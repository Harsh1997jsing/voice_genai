class AppException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code


class TwilioPurchaseException(AppException):
    pass


class InsufficientBalanceException(AppException):
    pass


class SubscriptionExpiredException(AppException):
    pass