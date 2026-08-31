import socket
import msgpack

class ServoController:
    PAN_CENTER = 94
    PAN_MIN = 4
    PAN_MAX = 184

    TILT_CENTER = 94
    TILT_MIN = 64
    TILT_MAX = 124

    def __init__(self, socket_path="/var/run/arduino-router.sock"):
        self.socket_path = socket_path
        self.msg_id = 0

        self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.client.connect(self.socket_path)

        self.unpacker = msgpack.Unpacker(raw=False)

        self.pan = self.PAN_CENTER
        self.tilt = self.TILT_CENTER

    def _rpc_call(self, method, *params):
        self.msg_id += 1

        request = [
            0,
            self.msg_id,
            method,
            list(params)
        ]
        self.client.sendall(msgpack.packb(request))

        while True:
            try:
                # Try to unpack complete msgpack-rpc object
                response = next(self.unpacker)
                break
            except StopIteration:
                # Incomplete data -> recv more from socket
                data = self.client.recv(1024)
                if not data:
                    raise ConnectionError(
                        "Arduino router closed the connection"
                    )
                self.unpacker.feed(data)

        msg_type, response_id, error, result = response

        if error is not None:
            raise RuntimeError(f"RPC error: {error}")
        if response_id != self.msg_id:
            raise RuntimeError(f"RPC ID mismatch: expected {self.msg_id}, got {response_id}")

        return result

    def set_cam(self, pan, tilt):
        self._rpc_call("set_cam", int(pan), int(tilt))
        self.pan = pan
        self.tilt = tilt

    def center(self):
        self._rpc_call("center")
        self.pan = self.PAN_CENTER
        self.tilt = self.TILT_CENTER

    def close(self):
        self.client.close()