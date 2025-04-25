import os
from urllib.parse import unquote
import pythoncom
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from scapy.all import load_layer
from scapy.sendrecv import AsyncSniffer

import wmi
from django.conf import settings

current_sniffer = None

@csrf_exempt
def start_sniffer(request):
    global current_sniffer

    if request.method == 'POST':
        input_file = request.FILES.get('input_file', None)
        input_interface = unquote(request.POST.get('input_interface', ''))
        #print('Decoded input_interface:', input_interface)
        output_mode = request.POST.get('output_mode')
        print(request.POST.get('output_file'))
        output_file = os.path.join(settings.MEDIA_ROOT,request.POST.get('output_file'))
        with open(output_file, 'w') as clear_file:
            clear_file.truncate(0)
        if input_file and input_interface:
            return JsonResponse({'error': 'Only one of input_file or input_interface should be provided.'}, status=400)
        if not input_file and not input_interface:
            return JsonResponse({'error': 'Either input_file or input_interface must be provided.'}, status=400)

        load_layer('tls')
        if input_file:
            file_path = os.path.join(settings.MEDIA_ROOT, input_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in input_file.chunks():
                    destination.write(chunk)
            input_file_copy = file_path
        else:
            input_file_copy = None
        current_sniffer = create_sniffer(input_file_copy, input_interface, output_mode, output_file)
        current_sniffer.start()

        return JsonResponse({'message': 'Sniffer started successfully', 'sniffer_status': 'running'})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)

@csrf_exempt
def stop_sniffer(request):
    global current_sniffer

    if request.method == 'POST':
        if current_sniffer:

            # if hasattr(current_sniffer, 'websocket_consumer'):
            #     current_sniffer.websocket_consumer.close_connection()
            # 停止嗅探器
            current_sniffer.stop()

            current_sniffer = None


            return JsonResponse({
                'message': 'Sniffer stopped successfully',
                'sniffer_status': 'stopped',
            })
        else:
            return JsonResponse({'error': 'No sniffer is currently running.'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)


def create_sniffer(input_file, input_interface, output_mode, output_file):
    from meter.session_utils import generate_session_class
    assert (input_file is None) ^ (input_interface == '')

    if input_file is not None:
        NewFlowSession = generate_session_class(output_mode, output_file, 'offline')
        #return AsyncSniffer(offline=input_file, filter='tcp port 443', prn=process_packet, session=NewFlowSession,store=False)
        return AsyncSniffer(offline=input_file, filter='tcp port 443', prn=None, session=NewFlowSession, store=False)
    else:
        NewFlowSession = generate_session_class(output_mode, output_file, 'online')
        return AsyncSniffer(iface=input_interface, filter='tcp port 443', prn=None, session=NewFlowSession, store=False)

def process_packet(packet):
    """提取缩略信息"""
    global captured_data

    if 'IP' in packet:
        summary = {
            'src_ip': packet['IP'].src,
            'dst_ip': packet['IP'].dst,
            'protocol': packet['IP'].proto,
            'length': len(packet)
        }
        captured_data.append(summary)

def get_interfaces(request):
    pythoncom.CoInitialize()
    w = wmi.WMI()
    interfaces = []
    for adapter in w.Win32_NetworkAdapter():
        if adapter.NetConnectionStatus == 2:  # 连接状态为已连接
            interfaces.append(adapter.Description)
    print(interfaces)
    return JsonResponse({'interfaces': interfaces})

if __name__ == '__main__':
    pass