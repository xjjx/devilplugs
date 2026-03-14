cmake_minimum_required(VERSION 3.22)
project(XjTF VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(juce)
add_subdirectory(chowdsp_wdf)

set(JUCE_VST3_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vst3sdk")
juce_set_vst2_sdk_path("${CMAKE_CURRENT_SOURCE_DIR}/VST_SDK_2.4")

# ── Helper: configure a JUCE plugin target ────────────────────────────────────
# Usage: xj_add_plugin(<target> <plugin_code> <lv2_slug> <src> [CHOWDSP])
#   CHOWDSP — optional flag, links chowdsp_wdf and adds its include path
function(xj_add_plugin TARGET PLUGIN_CODE LV2_SLUG SRC)
    cmake_parse_arguments(ARG "CHOWDSP" "" "" ${ARGN})

    juce_add_plugin(${TARGET}
        COMPANY_NAME                "XjDevilPlugs"
        PLUGIN_MANUFACTURER_CODE    Ystu
        PLUGIN_CODE                 ${PLUGIN_CODE}
        FORMATS                     LV2 VST VST3
        PRODUCT_NAME                ${TARGET}
        IS_SYNTH                    FALSE
        NEEDS_MIDI_INPUT            FALSE
        NEEDS_MIDI_OUTPUT           FALSE
        IS_MIDI_EFFECT              FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        VST3_SDK_PATH               "${JUCE_VST3_SDK_PATH}"
        LV2URI                      "urn:xjdevilplugs:${LV2_SLUG}"
    )

    target_sources(${TARGET} PRIVATE ${SRC})

    target_compile_definitions(${TARGET}
        PUBLIC
            JUCE_WEB_BROWSER=0
            JUCE_USE_CURL=0
            JUCE_VST3_CAN_REPLACE_VST2=0
    )

    set(PRIVATE_LIBS juce::juce_audio_utils juce::juce_dsp)
    if(ARG_CHOWDSP)
        list(APPEND PRIVATE_LIBS chowdsp_wdf)
        target_include_directories(${TARGET} SYSTEM PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/chowdsp_wdf/include
        )
    endif()

    target_link_libraries(${TARGET}
        PRIVATE ${PRIVATE_LIBS}
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_lto_flags
            juce::juce_recommended_warning_flags
    )

    if(MSVC)
        target_compile_options(${TARGET} PRIVATE
            $<$<CONFIG:Release>:/O2 /arch:AVX>
        )
    else()
        target_compile_options(${TARGET} PRIVATE
            $<$<CONFIG:Release>:-O3 -msse4.1 -mfpmath=sse -ffast-math>
        )
    endif()
endfunction()

# ── Plugins ───────────────────────────────────────────────────────────────────
xj_add_plugin(XjTF  XjTF xjtf src/XjTF/PluginProcessor.cpp  CHOWDSP)
xj_add_plugin(XjTFL XTFL xjtfl src/XjTFL/PluginProcessor.cpp)
