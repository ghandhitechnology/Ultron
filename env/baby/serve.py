#!/usr/bin/env python3

import socket
import threading


def serve(port: int) -> None:
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(8)
    while True:
        conn, _ = sock.accept()
        try:
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        except OSError:
            pass
        finally:
            conn.close()


def main() -> None:
    threading.Thread(target=serve, args=(8080,), daemon=True).start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
