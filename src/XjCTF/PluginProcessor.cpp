#include "PluginProcessor.h"

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout
InputTransformerAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    // Mode: 0=A, 1=S, 2=N
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        "mode", "Mode",
        juce::StringArray { "A", "S", "N" }, 0));

    // Drive: how hard the signal hits the transformer core
    // 0.0 = no effect, 1.0 = nominal, 2.0 = heavy saturation
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "drive", "Drive",
        juce::NormalisableRange<float>(0.0f, 2.0f, 0.01f, 0.5f), 0.5f));

    // Trim: output level compensation (-6 to +6 dB)
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "trim", "Trim",
        juce::NormalisableRange<float>(-6.0f, 6.0f, 0.1f), 0.0f));

    return layout;
}

//==============================================================================
InputTransformerAudioProcessor::InputTransformerAudioProcessor()
    : AudioProcessor(BusesProperties()
        .withInput ("Input",  juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
}

//==============================================================================
void InputTransformerAudioProcessor::prepareToPlay(double sr, int)
{
    sampleRate = sr;
    const double pi2 = juce::MathConstants<double>::twoPi;

    // ── Pre/de-emphasis shelf coefficients ───────────────────────────────────
    // 1-pole LP coeff — higher freq = less LP = more HF emphasis
    coeffs.a_pre = std::exp(-pi2 * 18000.0 / sr);  // Mode A: gentle ~18kHz
    coeffs.s_pre = std::exp(-pi2 * 15000.0 / sr);  // Mode S: ~15kHz
    coeffs.n_pre = std::exp(-pi2 * 10000.0 / sr);  // Mode N: deeper ~10kHz

    // ── DC blocker ~5Hz (shared) ──────────────────────────────────────────────
    coeffs.dc = std::exp(-pi2 * 5.0 / sr);

    // ── Mode N: 2-pole resonant LF bump ~60Hz, Q=1.8, +3dB ──────────────────
    {
        const double f0    = 60.0;
        const double Q     = 1.8;
        const double A     = std::pow(10.0, 3.0 / 40.0);  // +3dB peak gain
        const double w0    = pi2 * f0 / sr;
        const double cos0  = std::cos(w0);
        const double alpha = std::sin(w0) / (2.0 * Q);

        const double a0    =  1.0 + alpha / A;
        coeffs.n_lf_b0     = (1.0 + alpha * A) / a0;
        coeffs.n_lf_b1     = (-2.0 * cos0)      / a0;
        coeffs.n_lf_b2     = (1.0 - alpha * A)  / a0;
        coeffs.n_lf_a1     = (-2.0 * cos0)      / a0;
        coeffs.n_lf_a2     = (1.0 - alpha / A)  / a0;
    }

    // ── Reset all state ───────────────────────────────────────────────────────
    modeA = ModeAState{};
    modeS = ModeSState{};
    modeN = ModeNState{};
}

//==============================================================================
void InputTransformerAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                                   juce::MidiBuffer&)
{
    processImpl(buffer);
}

void InputTransformerAudioProcessor::processBlock(juce::AudioBuffer<double>& buffer,
                                                   juce::MidiBuffer&)
{
    processImpl(buffer);
}

//==============================================================================
template <typename Sample>
void InputTransformerAudioProcessor::processImpl(juce::AudioBuffer<Sample>& buffer)
{
    juce::ScopedNoDenormals noDenormals;

    const int  numSamples  = buffer.getNumSamples();
    const int  numChannels = buffer.getNumChannels();
    const bool mono        = (numChannels == 1);

    Sample* inL = buffer.getWritePointer(0);
    Sample* inR = mono ? nullptr : buffer.getWritePointer(1);

    const int   mode       = static_cast<int>(*apvts.getRawParameterValue("mode"));
    const float driveParam = *apvts.getRawParameterValue("drive");
    const float trimParam  = *apvts.getRawParameterValue("trim");

    // Drive 0..2 -> internal sat drive 1..6 (exponential feel)
    const double drive = 1.0 + std::pow((double)driveParam / 2.0, 1.5) * 5.0;

    // Emphasis amount scales with drive param, not internal drive
    // At driveParam=0: emphAmount=0 (pre/de cancel perfectly, no sat character)
    // At driveParam=2: emphAmount=1 (strong HF emphasis into saturator)
    const double emphBase = (double)driveParam / 2.0;

    const double trim = std::pow(10.0, (double)trimParam / 20.0);

    // Dry/wet: driveParam=0 -> fully dry passthrough
    const double wet = juce::jlimit(0.0, 1.0, (double)driveParam);
    const double dry = 1.0 - wet;

    for (int n = 0; n < numSamples; ++n)
    {
        const double L = static_cast<double>(*inL);
        const double R = mono ? 0.0 : static_cast<double>(*inR);
        double outL = L, outR = R;

        switch (mode)
        {
            // ── Mode A: even harmonics, gentle ───────────────────────────────
            case 0:
            {
                const double emph = emphBase * 0.4;  // subtle emphasis

                // Pre-emphasis (HF boost before sat)
                double fL = preEmphasis(L, modeA.preL, coeffs.a_pre, emph);
                double fR = preEmphasis(R, modeA.preR, coeffs.a_pre, emph);

                // Saturation
                fL = satModeA(fL, drive);
                fR = satModeA(fR, drive);

                // De-emphasis (restore flat response)
                fL = deEmphasis(fL, modeA.deL, coeffs.a_pre, emph);
                fR = deEmphasis(fR, modeA.deR, coeffs.a_pre, emph);

                // DC block
                outL = dcBlock(fL, modeA.dcL, modeA.dcHpL, coeffs.dc);
                outR = dcBlock(fR, modeA.dcR, modeA.dcHpR, coeffs.dc);
                break;
            }

            // ── Mode S: odd harmonics, punchy ────────────────────────────────
            case 1:
            {
                const double emph = emphBase * 0.55; // moderate emphasis

                double fL = preEmphasis(L, modeS.preL, coeffs.s_pre, emph);
                double fR = preEmphasis(R, modeS.preR, coeffs.s_pre, emph);

                fL = satModeS(fL, drive);
                fR = satModeS(fR, drive);

                fL = deEmphasis(fL, modeS.deL, coeffs.s_pre, emph);
                fR = deEmphasis(fR, modeS.deR, coeffs.s_pre, emph);

                outL = dcBlock(fL, modeS.dcL, modeS.dcHpL, coeffs.dc);
                outR = dcBlock(fR, modeS.dcR, modeS.dcHpR, coeffs.dc);
                break;
            }

            // ── Mode N: aggressive even harmonics + LF resonance ─────────────
            case 2:
            {
                const double emph = emphBase * 0.7;  // strongest emphasis

                // LF resonant bump (Marinair transformer core character)
                double fL = biquadTDF2(L, modeN.lf1L, modeN.lf2L,
                                       coeffs.n_lf_b0, coeffs.n_lf_b1, coeffs.n_lf_b2,
                                       coeffs.n_lf_a1, coeffs.n_lf_a2);
                double fR = biquadTDF2(R, modeN.lf1R, modeN.lf2R,
                                       coeffs.n_lf_b0, coeffs.n_lf_b1, coeffs.n_lf_b2,
                                       coeffs.n_lf_a1, coeffs.n_lf_a2);

                // Pre-emphasis (deeper shelf — more HF saturation)
                fL = preEmphasis(fL, modeN.preL, coeffs.n_pre, emph);
                fR = preEmphasis(fR, modeN.preR, coeffs.n_pre, emph);

                fL = satModeN(fL, drive);
                fR = satModeN(fR, drive);

                fL = deEmphasis(fL, modeN.deL, coeffs.n_pre, emph);
                fR = deEmphasis(fR, modeN.deR, coeffs.n_pre, emph);

                outL = dcBlock(fL, modeN.dcL, modeN.dcHpL, coeffs.dc);
                outR = dcBlock(fR, modeN.dcR, modeN.dcHpR, coeffs.dc);
                break;
            }

            default:
                outL = L;
                outR = R;
                break;
        }

        // Dry/wet blend + trim
        *inL++ = static_cast<Sample>((dry * L + wet * outL) * trim);
        if (!mono)
            *inR++ = static_cast<Sample>((dry * R + wet * outR) * trim);
    }
}

//==============================================================================
juce::AudioProcessorEditor* InputTransformerAudioProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

void InputTransformerAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void InputTransformerAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml != nullptr && xml->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xml));
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new InputTransformerAudioProcessor();
}
