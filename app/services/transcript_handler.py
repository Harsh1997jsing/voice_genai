# transcript_handler

async def handle_transcripts(
    stt_ws,
    tts_ws_holder: dict,
    tts_lock: asyncio.Lock,
    stt_control: dict,
    tts_state: dict,
    vad_state: dict,
    call_log: structlog.BoundLogger,
    user_id: int,
    send_audio_to_twilio,       # callable
    get_tts_pump_task,          # callable → current task or None
    set_tts_pump_task,
    conversation_history: list,
    dynamic_system_prompt          # callable(task)
):
    speculative_task = None
    last_partial = ""
    # FIX #4: use a dict instead of nonlocal vars to avoid closure race condition
    speculative_holder: dict = {"query": "", "result": None}


    async for result in receive_transcript(stt_ws):
        try:
            # ── PARTIAL ────────────────────────────────────────────────────
            if result["type"] == "partial":
                partial_text = result["text"].strip()

                if is_low_value(partial_text):
                    continue
                if not is_stable_transcript(partial_text):
                    continue
                if partial_text == last_partial:
                    continue

                last_partial = partial_text

                if speculative_task and not speculative_task.done():
                    speculative_task.cancel()

                query_for_embedding = normalize_query(
                                        extract_real_query(partial_text)
                                    )
                call_log.debug("speculative_queued", query=query_for_embedding)

                speculative_task = asyncio.create_task(
                    _speculative_search(
                        query=query_for_embedding,
                        user_id=user_id,
                        result_holder=speculative_holder,
                        debounce=SPECULATIVE_DEBOUNCE_SEC,
                        call_log=call_log,
                    )
                )
                continue

            # ── FINAL ──────────────────────────────────────────────────────
            if result["type"] == "final":
                user_text = result["text"].strip()

                END_CALL_PHRASES = [
                    "bye",
                    "goodbye",
                    "not interested",
                    "thank you",
                    "call later",
                    "stop calling",
                    "i'm busy",
                ]

                # if any(
                #     phrase in user_text.lower()
                #     for phrase in END_CALL_PHRASES
                # ):

                #     call_log.info(
                #         "end_call_phrase_detected",
                #         text=user_text
                #     )

                #     tts_state["end_call"] = True

                if any(
                    phrase in user_text.lower()
                    for phrase in END_CALL_PHRASES
                ):

                    call_log.info(
                        "end_call_phrase_detected",
                        text=user_text
                    )

                    call_state["ending"] = True

                    # stop current AI speaking immediately
                    tts_state["speaking"] = False
                    stt_control["paused"] = True

                    # optional goodbye message
                    async with tts_lock:
                        tts_ws = tts_ws_holder.get("ws")

                    if tts_ws:
                        try:
                            await send_text(
                                connection=tts_ws,
                                text="Thank you for calling. Goodbye."
                            )
                            await flush_stream(tts_ws)

                            # small delay so caller hears goodbye
                            await asyncio.sleep(1.5)

                        except Exception as e:
                            call_log.warning(
                                "goodbye_tts_failed",
                                error=str(e)
                            )

                    return

                if not user_text:
                    continue

                if is_low_value(user_text):
                    call_log.info("skipped_low_value", text=user_text)
                    if speculative_task and not speculative_task.done():
                        speculative_task.cancel()
                    speculative_task = None
                    last_partial = ""
                    continue

                stt_final_ts = time.perf_counter()
                call_log.info("stt_final", text=user_text)



                conversation_history.append({
                    "speaker": "user",
                    "text": user_text,
                    "timestamp": time.time(),
                })

                async with tts_lock:
                    tts_ws = tts_ws_holder.get("ws")

                if tts_ws is None:
                    call_log.warning("tts_ws_missing_on_final")
                    continue

                tts_state["speaking"] = True
                stt_control["paused"] = True

                trace_id = (
                    f"{tts_state.get('call_trace', 'call')}"
                    f"-{tts_state.get('turn', 0)}"
                )

                try:
                    docs = None
                    query_for_retrieval = normalize_query(
                                                extract_real_query(user_text)
                                            )

                    # Strategy 1: word-overlap cache hit
                    if (
                        speculative_holder["result"] is not None
                        and is_similar_query(query_for_retrieval, speculative_holder["query"])
                    ):
                        docs = speculative_holder["result"]
                        call_log.info("rag_cache_hit", strategy="speculative_reuse")

                    # Strategy 2: wait briefly for in-flight speculative
                    if not docs and speculative_task is not None:
                        try:
                            await asyncio.wait_for(asyncio.shield(speculative_task), timeout=0.3)
                            if speculative_holder["result"]:
                                docs = speculative_holder["result"]
                                call_log.info("rag_cache_hit", strategy="inflight_speculative")
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            call_log.debug("speculative_timed_out")

                    # Strategy 3: fresh retrieval
                    if not docs:
                        call_log.info("rag_fresh_retrieval", query=query_for_retrieval)
                        docs = await search_kb_async(
                            query=query_for_retrieval, user_id=user_id
                        )

                    docs = (docs or [])[:3]
                    context = "\n".join(docs)

                    retrieve_ms = (time.perf_counter() - stt_final_ts) * 1000
                    call_log.info(
                        "latency",
                        trace_id=trace_id,
                        stage="context_ready",
                        ms=round(retrieve_ms, 1),
                    )

                    buffer = ""
                    first_flush_done = False
                    sent_first_chunk = False

                    # Re-acquire tts_ws under lock before streaming
                    async with tts_lock:
                        tts_ws = tts_ws_holder.get("ws")

                    if tts_ws is None:
                        call_log.warning("tts_ws_gone_before_stream")
                        continue

                    assistant_response = ""

                    # ── LLM stream with hard timeout ───────────────────────
                    stream = stream_llm(
                        query=user_text,
                        context=context,
                        trace_id=trace_id,
                        user_id=user_id,
                        dynamic_system_prompt=dynamic_system_prompt,
                    )

                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                anext(stream),
                                timeout=LLM_STREAM_TIMEOUT_SEC
                            )
                        except StopAsyncIteration:
                            break

                        assistant_response += chunk 
                        if not sent_first_chunk:
                            first_chunk_ms = (
                                time.perf_counter() - stt_final_ts
                            ) * 1000

                            call_log.info(
                                "latency",
                                trace_id=trace_id,
                                stage="first_llm_chunk",
                                ms=round(first_chunk_ms, 1),
                            )

                            sent_first_chunk = True

                        buffer += chunk

                        await send_text(
                            connection=tts_ws,
                            text=chunk,
                        )

                        if not first_flush_done:
                            await flush_stream(tts_ws)
                            first_flush_done = True
                            buffer = ""

                        elif len(buffer) > 120:
                            await flush_stream(tts_ws)
                            buffer = ""

                    if buffer.strip():
                        await flush_stream(tts_ws)

                    conversation_history.append({
                        "speaker": "assistant",
                        "text": assistant_response,
                        "timestamp": time.time(),
                    }) 

                except asyncio.TimeoutError:
                    call_log.error("llm_stream_timeout", trace_id=trace_id)

                except Exception as e:
                    call_log.error("turn_error", trace_id=trace_id, error=str(e))

                finally:
                    tts_state["speaking"] = False
                    stt_control["paused"] = False
                    tts_state["turn"] += 1

                    speculative_task = None
                    last_partial = ""
                    speculative_holder["query"] = ""
                    speculative_holder["result"] = None

                    
        except Exception as e:
            # Outer safety net — log and keep the loop alive
            call_log.error("transcript_loop_error", error=str(e))
            continue



async def _speculative_search(
    query: str,
    user_id: int,
    result_holder: dict,
    debounce: float,
    call_log,
) -> None:
    """
    Waits `debounce` seconds, then runs KB search and writes into
    result_holder. Cancelled cleanly if a newer partial arrives.
    Defined outside handle_transcripts to avoid nonlocal race conditions
    when multiple tasks are created in rapid succession.
    """
    try:
        await asyncio.sleep(debounce)
        res = await search_kb_async(query=query, user_id=user_id)
        result_holder["query"] = query
        result_holder["result"] = res
        call_log.debug("speculative_done", query=query)
    except asyncio.CancelledError:
        pass  # newer partial superseded us — expected