# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(
    ['main.py'],  # Main script for your application
    pathex=[r"c:\Users\timsp\OneDrive\Desktop\TelePrompt 2\teleprompt"],  # Project root path
    binaries=[],
    datas=[
        # Include Sentence Transformer Models
        ('local_models', 'local_models'),

        # Include Tesseract OCR Engine and Tessdata
        ('tesseract', 'tesseract'),

        # Include Poppler PDF Rendering Binaries
        ('poppler', 'poppler'),

        # Include Google Cloud Credentials File
        ('hazel-sky-450118-p6-0ffc677fdd97.json', '.'),

        # Include TelePrompt Configuration File
        ('config/teleprompt_config.yaml', 'config'),

        # Include Voicemeeter Settings XML Files
        ('load_these_voicemeeter_settings.xml', '.'),
        ('teleprompt_voicemeter_settings.xml', '.'),

        # Include Application Icon Files
        ('my_tray_icon.png', '.'),
        ('logo/logo.png', 'logo'),
        ('teleprompt_icon.ico', '.'),

        # Include Voicemeeter Installer
        ('voicemeter_installer/VoicemeeterSetup.exe', 'voicemeter_installer'),

        # Include Voicemeeter Remote API DLL (64-bit)
        ('VoicemeeterRemote64.dll', '.'),
    ],
    hiddenimports=['audio_processing'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=True,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, name='pyz_main')

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TelePrompt',  # Application executable name
    debug=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for a windowed application (no console)
    windowed=True,
    icon='teleprompt_icon.ico',  # Path to the icon file
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TelePrompt',  # Output folder name
)
