from .backend import GuestBackend, GuestHandle


def host_confirm_root(backend: GuestBackend, guest: GuestHandle, username: str) -> bool:
    return backend.confirm_root(guest, username)
