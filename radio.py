import scipy.signal as signal
import numpy as np
import rtlsdr
import argparse
from threading import Thread, Lock
from Queue import Queue
import pyaudio
import sys
import time

SUBCARRIER_LOW = 67.5e3
SUBCARRIER_HIGH = 92e3
SAMPLE_RATE = 240e3
GAIN = 36
BLOCKSIZE = 2 * 256e3
AUDIO_FS = 48000
STEREO_DEMODULATE_CARRIER = 38e3


RADIO = rtlsdr.RtlSdr()
RADIO.sample_rate = SAMPLE_RATE
RADIO.gain = GAIN
RADIO_LOCK = Lock()

RADIO_LOCK = Lock()

AUDIO_QUEUE = Queue()

AUDIO = pyaudio.PyAudio()
AUDIO_LOCK = Lock()

class AsyncRadioReceiveDemodulateProcess(Thread):
    def __init__(self, carrier, scheme, whichSubband):
        Thread.__init__(self)
        RADIO_LOCK.acquire()
        self.carrier = carrier
        self.scheme = scheme
        self.whichSubband = whichSubband

    def run(self):

        def dual_sideband_am_demodulate(sample):
            f_0 = self.carrier
            t_max = BLOCKSIZE / SAMPLE_RATE
            t = np.linspace(0, t_max, len(sample))
            subcarrier_demodulator = np.exp(-2j * np.pi * f_0 * t)
            subcarrier_demodulated = sample * subcarrier_demodulator
            return subcarrier_demodulated

    
        def fm_demodulate(sample):
            first = sample[1:]
            second = np.conj(sample[0:-1])
            phase_product = first * second
            phase_difference = np.angle(phase_product) / (2 * np.pi)
            derivative = phase_difference / SAMPLE_RATE
            return derivative

        def scale(sample):
            max_mono_amplit = max(sample)
            min_mono_amplit = min(sample)
            center = (max_mono_amplit + min_mono_amplit) / 2.0

            return (sample - center) * \
                (2.0 / (max_mono_amplit - min_mono_amplit))
            

        def demodulateCallback(sample, context):
            queue_data = None
            if self.scheme == 'AM':
                demodulated = dual_sideband_am_demodulate(sample)
                
                # audio filter
                h = signal.firwin(128, 5e3, nyq=SAMPLE_RATE / 2.0)
                am_filtered = signal.fftconvolve(demodulated, h)

                downsampled_am = am_filtered[0::5]
                queue_data = scale(downsampled_am)

            else:
                derivative = fm_demodulate(sample)
                if not self.whichSubband:

                    # 1. Low pass filter at 16 kHz
                    h = signal.firwin(128, 16e3, nyq=SAMPLE_RATE / 2.0)
                    mono_filtered = signal.fftconvolve(derivative, h)
                    
                    # 2. Downsample to a readable frequency
                    downsampled_mono = mono_filtered[0::5]

                    #3. Scale signal to between -1 and 1
                    queue_data = scale(downsampled_mono)

                else:
                    f_0 = SUBCARRIER_LOW

                    if self.whichSubband == 'high':
                        f_0 = SUBCARRIER_HIGH
                        
                    t_max = BLOCKSIZE / SAMPLE_RATE
                    t = np.linspace(0, t_max, len(derivative))
                    subcarrier_demodulator = np.exp(-2j * np.pi * f_0 * t)
                    subcarrier_demodulated = derivative * subcarrier_demodulator

                    subcarrier_passband_filter = signal.firwin(128, cutoff=7.5e3,  
                                                               nyq=SAMPLE_RATE/2)
                    subcarrier_filtered = signal.fftconvolve(
                            subcarrier_demodulated, 
                            subcarrier_passband_filter)

                    subcarrier_fm_demodulate = fm_demodulate(subcarrier_filtered)

                    downsampled_subcarrier = subcarrier_fm_demodulate[0::5]

                    scaled_subcarrier = scale(downsampled_subcarrier)

                    audio_lowpass_h = signal.firwin(128, cutoff=7.5e3, nyq=AUDIO_FS/2)

                    queue_data = signal.fftconvolve(audio_lowpass_h, scaled_subcarrier)

            AUDIO_QUEUE.put(queue_data)

        RADIO.read_samples_async(demodulateCallback,
                        num_samples=BLOCKSIZE)

    def shutdown(self):
        RADIO.cancel_read_async()
        RADIO_LOCK.release()


class AsyncPlayAudioProcess(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):

        AUDIO_LOCK.acquire()
        stream = AUDIO.open(format = pyaudio.paFloat32,
                        channels = 1,
                        rate = AUDIO_FS,
                        output = True)

        while True:
            data = AUDIO_QUEUE.get()
            stream.write(data.astype(np.float32).tostring())
        stream.close()

    def shutdown(self):
        AUDIO_LOCK.release()
        
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='command-line based VHF radio')
    parser.add_argument('carrier_freq', metavar='freq', type=float,
                        help='carrier frequency for station')
    parser.add_argument('-sa', '--subA', action='store_true',
                        help='tune to 67.5 kHz subcarrier')
    parser.add_argument('-sb', '--subB', action='store_true',
                        help='tune to 92 kHZ subcarrier')
    parser.add_argument('-AM','--AM', action='store_true',
                        help='set demodulation scheme to dual-sideband AM')

    args = parser.parse_args()

    carrier_frequency = args.carrier_freq

    subcarrier_frequency = None
    if args.subA:
        subcarrier_frequency = 'low'
    elif args.subB:
        subcarrier_frequency = 'high'


    demodulation_scheme = 'FM'
    if args.AM:
        demodulation_scheme = 'AM'


    RADIO.center_freq = carrier_frequency

    radio_process = AsyncRadioReceiveDemodulateProcess(carrier_frequency,
                                                    demodulation_scheme, 
                                                    subcarrier_frequency)
    radio_process.daemon = True

    audio_process = AsyncPlayAudioProcess()
    audio_process.daemon = True

    try:
        radio_process.start()
        audio_process.start()
        while True:
            time.sleep(5)
    except Exception as e:
        radio_process.shutdown()
        audio_process.shutdown()
        RADIO.close()
        print 'Quitting...'
