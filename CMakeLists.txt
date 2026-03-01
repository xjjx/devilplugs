cmake_minimum_required(VERSION 3.22)

project(XjTF VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE
add_subdirectory(juce)
add_subdirectory(chowdsp_wdf)

# VST3 SDK - tell JUCE where to find it
set(JUCE_VST3_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vst3sdk")

# Point JUCE to the VST2 SDK
juce_set_vst2_sdk_path("${CMAKE_CURRENT_SOURCE_DIR}/VST_SDK_2.4")

juce_add_plugin(XjTF
    COMPANY_NAME "XjDevilPlugs"
    PLUGIN_MANUFACTURER_CODE Ystu
    PLUGIN_CODE XjTF
    FORMATS LV2 VST VST3
    PRODUCT_NAME "XjTF"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    VST3_SDK_PATH "${JUCE_VST3_SDK_PATH}"
    LV2URI "urn:xjdevilplugs:xjtf"
)

target_include_directories(XjTF SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/chowdsp_wdf/include
)

target_sources(XjTF
    PRIVATE
        src/XjTF/PluginProcessor.cpp
)

target_compile_definitions(XjTF
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(XjTF
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        chowdsp_wdf
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
