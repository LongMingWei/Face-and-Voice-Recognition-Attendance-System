import pveagle
from pveagle import EagleProfile
from pvrecorder import PvRecorder

access_key = "/cxYMcgSAw1EMF0XPQT2tXUKHoHqXSCvByLxTjtmMjHfwgkN/ypnag=="

eagle_profiler = pveagle.create_profiler(access_key=access_key)
recorder = PvRecorder(device_index=-1, frame_length=eagle_profiler.min_enroll_samples)
recorder.start()
enroll_percentage = 0.0
while enroll_percentage < 100.0:
    audio_frame = recorder.read()
    enroll_percentage, feedback = eagle_profiler.enroll(audio_frame)
    print(enroll_percentage)
recorder.stop()

speaker_profile = eagle_profiler.export()
voicebyte = speaker_profile.to_bytes()
eagle_profiler.delete()
recorder.delete()


speaker_profile = EagleProfile.from_bytes(voicebyte)
eagle = pveagle.create_recognizer(access_key=access_key, speaker_profiles=[speaker_profile])
recognizer_recorder = PvRecorder(device_index=-1, frame_length=eagle.frame_length)
recognizer_recorder.start()

for i in range(1000):
    audio_frame = recognizer_recorder.read()
    print(eagle.process(audio_frame)[0])
recognizer_recorder.stop()
eagle.delete()
recognizer_recorder.delete()
