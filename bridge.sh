kill $(ps aux | grep 'import socket' | grep -v grep | awk '{print $2}') 2>/dev/null
python3 -c "
import socket, threading

def pipe(src, dst):
    try:
        while True:
            data = src.recv(4096)
            if not data: break
            dst.sendall(data)
    except: pass
    finally:
        src.close(); dst.close()

def start_bridge(port):
    def handle_client(c):
        try:
            target = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target.connect(('127.0.0.1', port))
            threading.Thread(target=pipe, args=(c, target), daemon=True).start()
            threading.Thread(target=pipe, args=(target, c), daemon=True).start()
        except:
            c.close()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Force reuse
    s.bind(('localhost', port))
    s.listen(5)
    print(f'Bridge Active on Port {port}')
    while True:
        client_sock, addr = s.accept()
        handle_client(client_sock)

threading.Thread(target=start_bridge, args=(2000,), daemon=True).start()
start_bridge(2001)
"