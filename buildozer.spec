[app]
# (str) Title of your application
title = Direct IP Voice
# (str) Package name
package.name = direct_ip_voice
# (str) Package domain (needed for android)
package.domain = org.example
# (str) Source code where the main.py is located
source.dir = .
# (str) Application entry point
# Buildozer will use main.py by default; ensure main.py launches the Kivy app

# (list) Application requirements - minimal to speed up build
requirements = python3,kivy

# (str) Supported orientation (landscape, portrait or all)
orientation = portrait

# (list) Permissions - only essentials
android.permissions = RECORD_AUDIO,INTERNET,MODIFY_AUDIO_SETTINGS

# (int) Target API
android.api = 33
# (int) Minimum API
android.minapi = 21

# (str) Package version
version = 0.1

# (str) Android architectures - limit to 32-bit to speed build
# Note: Use only armeabi-v7a for faster builds; for modern devices include arm64 if needed
android.arch = armeabi-v7a

# (str) Android entrypoint
# (others)

[buildozer]
# reduce verbosity and speed up incremental builds
log_level = 1
warn_on_root = 0
