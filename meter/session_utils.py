# session_utils.py
from meter.flow_session import FlowSession

def generate_session_class(output_mode, output_file, input_mode):
    return type('NewFlowSession', (FlowSession,), {
        'output_mode': output_mode,
        'output_file': output_file,
        'input_mode': input_mode,
    })
