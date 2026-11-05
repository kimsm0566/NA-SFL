import socket
import time


class BandwidthLimiter:
    def __init__(self, upload_speed_bytes_per_s=None, download_speed_bytes_per_s=None):
        self.upload_delay_per_byte = 1 / upload_speed_bytes_per_s if upload_speed_bytes_per_s else 0
        self.download_delay_per_byte = 1 / download_speed_bytes_per_s if download_speed_bytes_per_s else 0

    def sendall(self, sock, data):
        start_time = time.time()
        sock.sendall(data)
        elapsed_time = time.time() - start_time
        sleep_time = len(data) * self.upload_delay_per_byte - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    def recv(self, sock, buffer_size):
        start_time = time.time()
        data = sock.recv(buffer_size, socket.MSG_WAITALL)
        elapsed_time = time.time() - start_time
        sleep_time = len(data) * self.download_delay_per_byte - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        return data


# Sample usage:
if __name__ == '__main__':
    upload_speed_kb_per_s = float(input("Enter max upload speed (KB/s): "))
    download_speed_kb_per_s = float(input("Enter max download speed (KB/s): "))
    limiter = BandwidthLimiter(upload_speed_kb_per_s, download_speed_kb_per_s)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(('localhost', 10000))

        # Send data using limiter's sendall method
        message = 'Hello, server! This is device.'
        print('Sending:', message)
        limiter.sendall(sock, message.encode())

        # Receive a response using limiter's recv method
        response = limiter.recv(sock, 1024)
        print('Received:', response.decode())
