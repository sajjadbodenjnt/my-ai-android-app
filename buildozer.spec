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
entrypoint = direct_ip_call.py

# (list) Application requirements
requirements = python3,kivy,pyjnius

# (str) Supported orientation (landscape, portrait or all)
orientation = portrait

# (list) Permissions
android.permissions = RECORD_AUDIO, INTERNET, ACCESS_WIFI_STATE, CHANGE_WIFI_STATE, MODIFY_AUDIO_SETTINGS, CAMERA, android.permission.HIGH_PRIORITY_BACKGROUND_WORK

# (int) Target API
android.api = 33
# (int) Minimum API
android.minapi = 21
# (str) Android NDK
#android.ndk = 21b

# (str) Package version
version = 0.1

# (str) Android archs
android.arch = armeabi-v7a

# (str) Android entrypoint
# (others)

[buildozer]
log_level = 2
warn_on_root = 1
