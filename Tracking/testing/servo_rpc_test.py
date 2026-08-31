import socket
import msgpack
import time

SOCKET_PATH = "/var/run/arduino-router.sock" # where the router listens
msg_id = 0 # protocol bookeeping - increment request IDs


def rpc_call(method, *params):
    global msg_id
    msg_id += 1
	
	# msgpack-rpc request - [type (0=request), ID, name, args]
    request = [
        0,
        msg_id,
        method,
        list(params)
    ]
    payload = msgpack.packb(request)
    
	# Open local Unix-domain socket to arduino-router
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(SOCKET_PATH)
        client.sendall(payload)

        # Stream socket may return response in multiple chunks,
        # so accumulate data until the whole object exists
        unpacker = msgpack.Unpacker(raw=False)
        while True:
            data = client.recv(1024)
            if not data:
                raise ConnectionError(
                    "Router closed connection before sending a complete RPC response"
                )
            unpacker.feed(data) # append raw bytes to unpacker's buffer

            try:
                # Try to unpack complete msgpack-rpc object
                response = next(unpacker)
                break
            except StopIteration:
                # Buffer does not contain full object
                continue

    # MessagePack-RPC response - [type (1=response), ID, error (None=OK), result]
    msg_type, response_id, error, result = response

    if error is not None:
        raise RuntimeError(f"RPC error: {error}")
    if response_id != msg_id:
        raise RuntimeError(f"RPC response ID mismatch: expected {msg_id}, got {response_id}")

    return result


print("Center")
rpc_call("set_cam", 94, 94)
time.sleep(2)

print("Pan right")
rpc_call("set_cam", 60, 94) # pan right
time.sleep(2)

print("Pan left")
rpc_call("set_cam", 130, 94) # pan left
time.sleep(2)

print("Tilt down")
rpc_call("set_cam", 94, 75) # tilt down
time.sleep(2)

print("Tilt up")
rpc_call("set_cam", 94, 110) # tilt up
time.sleep(2)

print("Center")
rpc_call("set_cam", 94, 94)

print("Done")