import aifc
import numpy as np
from scipy.io.wavfile import write


soundIn = aifc.open('beat_dummy.aiff')
readIn = soundIn.readframes(soundIn.getnframes())
y = np.fromstring(readIn, numpy.short).byteswap()

fs = soundIn.getframerate()
nbBytes = soundIn.getsampwidth()

listy = list(y)
evenlisty = listy[::2]
evenarrayy = np.asarray(evenlisty)

write('test2.wav',fs,evenarrayy)