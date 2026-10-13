from .model_main import UrbanSeg
MAIN_ARCHS = ('urbanges',)
CMP_ARCHS = {}

def build_model(arch, num_classes=9, img_size=1536, encoder_type='dinov3', probe_only=False):
    if arch in (None, 'urbanges'):
        return UrbanSeg(num_classes=num_classes, img_size=img_size, encoder_type=encoder_type, probe_only=probe_only)
    raise KeyError(f"未知 arch: {arch}；可选: {', '.join(MAIN_ARCHS)}")
