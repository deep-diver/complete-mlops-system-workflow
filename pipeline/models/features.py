IMAGE_KEY = 'image'
LABEL_KEY = 'label'

def transformed_name(key: str) -> str:
  """Generate the name of the transformed feature from original name."""
  return key + '_xf'
