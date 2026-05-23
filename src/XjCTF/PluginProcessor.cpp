#include "PluginProcessor.h"

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout
InputTransformerAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    // Mode: 0=S, 1=A, 2=N
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        "mode", "Mode",
        juce::StringArray { "A", "S", "N" }, 0));

    // Drive: how hard the signal hits the transformer core
    // 0.0 = bypass, 1.0 = nominal, beyond that = saturation
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "drive", "Drive",
        juce::NormalisableRange<float>(0.0f, 2.0f, 0.01f, 0.5f), 0.5f));

    // Trim: output compensation (-6 to +6 dB)
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

    // ── Mode A ───────────────────────────────────────────────────────────────
    // HF 1-pole lowpass ~18kHz (gentle rolloff)
    coeffs.mode_s_hf = std::exp(-pi2 * 18000.0 / sr);
    // DC blocker ~5Hz
    coeffs.mode_s_dc = std::exp(-pi2 * 5.0 / sr);

    // ── Mode S ───────────────────────────────────────────────────────────────
    // HF 1-pole lowpass ~15kHz
    coeffs.mode_a_hf = std::exp(-pi2 * 15000.0 / sr);
    coeffs.mode_a_dc = std::exp(-pi2 * 5.0 / sr);

    // ── Mode N ───────────────────────────────────────────────────────────────
    // 2-pole resonant LF bump ~60Hz, Q~1.8
    // Using bilinear-transformed resonator
    {
        const double f0  = 60.0;
        const double Q   = 1.8;
        const double w0  = pi2 * f0 / sr;
        const double cos0 = std::cos(w0);
        const double sin0 = std::sin(w0);
        const double alpha = sin0 / (2.0 * Q);

        // Peaking EQ +3dB at f0 for the transformer bump character
        const double A    = std::pow(10.0, 3.0 / 40.0); // +3dB
        const double b0   =  1.0 + alpha * A;
        const double b1   = -2.0 * cos0;
        const double b2   =  1.0 - alpha * A;
        const double a0   =  1.0 + alpha / A;
        const double a1   = -2.0 * cos0;
        const double a2   =  1.0 - alpha / A;

        coeffs.mode_n_lf_b0 = b0 / a0;
        coeffs.mode_n_lf_b1 = b1 / a0;
        coeffs.mode_n_lf_b2 = b2 / a0;
        coeffs.mode_n_lf_a1 = a1 / a0;
        coeffs.mode_n_lf_a2 = a2 / a0;
    }
    // HF 1-pole lowpass ~12kHz (Marinair bandwidth limit)
    coeffs.mode_n_hf    = std::exp(-pi2 * 12000.0 / sr);
    // Extra HF shelf cut at ~8kHz (additional darkness)
    coeffs.mode_n_shelf = std::exp(-pi2 * 8000.0  / sr);
    coeffs.mode_n_dc    = std::exp(-pi2 * 5.0 / sr);

    // ── Reset all state ───────────────────────────────────────────────────────
    mode_s = SState{};
    mode_a = AState{};
    mode_n = NState{};
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

    const int numSamples  = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();
    const bool mono       = (numChannels == 1);

    Sample* inL = buffer.getWritePointer(0);
    Sample* inR = mono ? nullptr : buffer.getWritePointer(1);

    const int   mode  = static_cast<int>(*apvts.getRawParameterValue("mode"));
    const float driveParam = *apvts.getRawParameterValue("drive");
    const float trimParam  = *apvts.getRawParameterValue("trim");

    // Drive maps 0..2 → internal drive 1.0..6.0 (exponential feel)
    const double drive = 1.0 + std::pow((double)driveParam / 2.0, 1.5) * 5.0;
    const double trim  = std::pow(10.0, (double)trimParam / 20.0);

    for (int n = 0; n < numSamples; ++n)
    {
        double L = static_cast<double>(*inL);
        double R = mono ? 0.0 : static_cast<double>(*inR);
        double outL = L, outR = R;

        // ── Blend amount: drive 0 = full dry, drive 1+ = increasing wet ──────
        // At drive=0 (driveParam=0) we pass through cleanly
        const double wet = juce::jlimit(0.0, 1.0, (double)driveParam);
        const double dry = 1.0 - wet;

        switch (mode)
        {
            // ── Mode A ───────────────────────────────────────────────────────
            case 0:
            {
                // 1. HF rolloff (pre-saturation, transformer bandwidth)
                mode_s.hfL = mode_s.hfL + (1.0 - coeffs.mode_s_hf) * (L - mode_s.hfL);
                mode_s.hfR = mode_s.hfR + (1.0 - coeffs.mode_s_hf) * (R - mode_s.hfR);
                double fL = mode_s.hfL;
                double fR = mode_s.hfR;

                // 2. Even-harmonic saturation (very subtle at low drive)
                fL = sSat(fL, drive);
                fR = sSat(fR, drive);

                // 3. DC block
                outL = dcBlock(fL, mode_s.dcL, mode_s.dcHpL, coeffs.mode_s_dc);
                outR = dcBlock(fR, mode_s.dcR, mode_s.dcHpR, coeffs.mode_s_dc);
                break;
            }

            // ── Mode S ───────────────────────────────────────────────────────
            case 1:
            {
                // 1. HF rolloff (~15kHz, slightly darker than hardware)
                mode_a.hfL = mode_a.hfL + (1.0 - coeffs.mode_a_hf) * (L - mode_a.hfL);
                mode_a.hfR = mode_a.hfR + (1.0 - coeffs.mode_a_hf) * (R - mode_a.hfR);
                double fL = mode_a.hfL;
                double fR = mode_a.hfR;

                // 2. Odd-harmonic saturation (forward, punchy)
                fL = aSat(fL, drive);
                fR = aSat(fR, drive);

                // 3. DC block
                outL = dcBlock(fL, mode_a.dcL, mode_a.dcHpL, coeffs.mode_a_dc);
                outR = dcBlock(fR, mode_a.dcR, mode_a.dcHpR, coeffs.mode_a_dc);
                break;
            }

            // ── Mode N ───────────────────────────────────────────────────────
            case 2:
            {
                // Transposed Direct Form II biquad (stable)
                auto biquadDF2 = [&](double x, double& s1, double& s2) -> double
                {
                    double y = coeffs.mode_n_lf_b0 * x + s1;
                    s1 = coeffs.mode_n_lf_b1 * x - coeffs.mode_n_lf_a1 * y + s2;
                    s2 = coeffs.mode_n_lf_b2 * x - coeffs.mode_n_lf_a2 * y;
                    return y;
                };

                double fL = biquadDF2(L, mode_n.lf1L, mode_n.lf2L);
                double fR = biquadDF2(R, mode_n.lf1R, mode_n.lf2R);

                // 2. HF rolloff — two stages for steeper Marinair rolloff
                mode_n.hfL   = mode_n.hfL   + (1.0 - coeffs.mode_n_hf)    * (fL - mode_n.hfL);
                mode_n.shelfL = mode_n.shelfL + (1.0 - coeffs.mode_n_shelf) * (mode_n.hfL - mode_n.shelfL);
                mode_n.hfR   = mode_n.hfR   + (1.0 - coeffs.mode_n_hf)    * (fR - mode_n.hfR);
                mode_n.shelfR = mode_n.shelfR + (1.0 - coeffs.mode_n_shelf) * (mode_n.hfR - mode_n.shelfR);
                fL = mode_n.shelfL;
                fR = mode_n.shelfR;

                // 3. Aggressive even-harmonic tanh saturation
                fL = nSat(fL, drive);
                fR = nSat(fR, drive);

                // 4. DC block
                outL = dcBlock(fL, mode_n.dcL, mode_n.dcHpL, coeffs.mode_n_dc);
                outR = dcBlock(fR, mode_n.dcR, mode_n.dcHpR, coeffs.mode_n_dc);
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
