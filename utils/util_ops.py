import subprocess
import xml.etree.ElementTree


def get_num_devices():
  try:
    subprocess.check_output(['which', 'nvidia-smi']).decode('utf-8')
  except subprocess.CalledProcessError:
    # Only CPU available
    return 0

  xml_output = subprocess.check_output(['nvidia-smi', '-q', '-x']).decode(
    'utf-8')

  e = xml.etree.ElementTree.fromstring(xml_output)

  gpus = e.findall('gpu')

  return len(gpus)
