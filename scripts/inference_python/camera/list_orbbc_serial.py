from pyorbbecsdk import *

def list_all_devices():
    """
    列出所有连接的设备及其序列号
    """
    context = Context()
    device_list = context.query_devices()
    
    print(f"Available devices number : {device_list.get_count()}")
    for i in range(device_list.get_count()):
        device = device_list.get_device_by_index(i)
        
        device_info = device.get_device_info()
        name = device_info.get_name()
        serial_number = device_info.get_serial_number()
        vid = device_info.get_vid()
        pid = device_info.get_pid()
        print(f"Index: {i}, Name: {name}, Serial: {serial_number}, VID: {vid}, PID: {pid}")

if __name__ == "__main__":
    list_all_devices()