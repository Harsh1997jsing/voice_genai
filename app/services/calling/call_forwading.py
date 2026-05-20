
from app.core.twilio_clinet import client

def transfer_to_human(call_sid):
    client.calls(call_sid).update(
        twiml="""
        <Response>
            <Say>
                Connecting you to a human agent.
            </Say>
            <Dial>
                +919999999999
            </Dial>
        </Response>
        """
    )