from dashscope.audio.asr import *
from dashscope import MultiModalConversation
import dashscope
import asyncio
import os
import sounddevice as sd
import numpy as np
import threading
import time

import json
import base64
from typing import List, Optional


class Callback(RecognitionCallback):
    def __init__(self, text, frames, owner) -> None:
        self.temp_text = []
        self.text = text
        self.frames = frames
        self.owner = owner

    def on_open(self) -> None:
        print("RecognitionCallback open.")

    def on_close(self) -> None:
        print("RecognitionCallback close.")
        if self.temp_text:
            self.text.extend(self.temp_text)
            self.temp_text.clear()

    def on_complete(self) -> None:
        print("RecognitionCallback completed.")  # translation completed
        if self.temp_text:
            self.text.extend(self.temp_text)
            self.temp_text.clear()

    def on_error(self, message) -> None:
        print("RecognitionCallback task_id: ", message.request_id)
        print("RecognitionCallback error: ", message.message)
        if self.temp_text:
            self.text.extend(self.temp_text)
            self.temp_text.clear()

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if sentence is None:
            return

        if isinstance(sentence, list):
            sentence_list = sentence
        else:
            sentence_list = [sentence]

        for current in sentence_list:
            if "text" in current:
                print("RecognitionCallback text: ", current["text"])
                self.temp_text.append(current["text"])
                if RecognitionResult.is_sentence_end(current):
                    print(
                        "RecognitionCallback sentence end, request_id:%s, usage:%s"
                        % (result.get_request_id(), result.get_usage(current))
                    )
                    last_text = current.get("text", "")
                    if last_text:
                        self.text.append(last_text)
                    self.temp_text.clear()
            if "emo_tag" in current:
                print("RecognitionCallback emo_tag: ", current["emo_tag"])
                self.owner.emo_tag = current["emo_tag"]

    def reset(self) -> None:
        self.temp_text.clear()


class SpeechService:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        sample_rate = 16000

        # 资源存储变量
        self.frames: List[np.ndarray] = []  # 声音数组
        self.transcribed_text: List[str] = []  # 文本数组
        self.emo_tag = None

        with open(config_path, "r") as f:
            config = json.load(f)
        self.dashscope_api_key = config["DASHSCOPE_API_KEY"]
        dashscope.api_key = self.dashscope_api_key
        # 看了官方的api,目前就是支持用callback来进行流式调用,也可以不填这个参数一次性传递完，我选择阻塞的方法因为我的最终转换的text是prompt,这个prompt必须完整
        self.callback = Callback(self.transcribed_text, self.frames, self)
        self._recognition_started = False
        self.SpeechToTextClient = Recognition(
            model="paraformer-realtime-v2",
            format="pcm",
            sample_rate=sample_rate,
            heartbeat=True,
            callback=self.callback,
        )
        try:
            self.SpeechToTextClient.start()  # 流式的开始识别
            self._recognition_started = True
        except Exception as e:
            print(f"Error initializing recognition: {e}")
            self._recognition_started = False
        self.SpeakerClient = MultiModalConversation
        self.is_recording = False

        self.recording_thread = None

        # Audio recording parameters
        self.sample_rate = sample_rate
        self.channels = 1
        self.dtype = np.int16

    def start_recording(self):
        """多线程开启录音"""
        if self.is_recording:
            return False

        if not self._recognition_started:
            try:
                self.SpeechToTextClient.start()
                self._recognition_started = True
            except Exception as e:
                print(f"Error starting recognition: {e}")
                return False

        if self.callback:
            self.callback.reset()

        self.is_recording = True
        self.frames.clear()
        self.transcribed_text.clear()
        self.emo_tag = None

        try:
            # 录音的线程
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            if self._recognition_started:
                try:
                    self.SpeechToTextClient.stop()
                except Exception as stop_error:
                    print(f"Error stopping recognition after failure: {stop_error}")
                self._recognition_started = False
            return False

    def stop_recording(self):
        """把暂停的逻辑放到回调函数了,这个保留为主动暂停的函数，可能用不上的"""
        if not self.is_recording:
            return None

        self.is_recording = False

        if self.recording_thread:
            self.recording_thread.join()
            self.recording_thread = None

        if self._recognition_started:
            try:
                self.SpeechToTextClient.stop()
            except Exception as e:
                print(f"Error stopping recognition: {e}")
            self._recognition_started = False
        if not self.transcribed_text:
            return None
        return "".join(self.transcribed_text)

    def _record_audio(self):
        """Record audio using sounddevice"""
        try:
            # Record audio until stopped
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
            ) as stream:
                # 当这个结束的时候就没有结束了
                while self.is_recording:
                    sd.sleep(100)  # Sleep in milliseconds
        except Exception as e:
            print(f"Error recording audio: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        """这是核心的回调函数,sounddevice每录制到一个CHUNK,就会调用一次这个函数。"""
        if not self.is_recording:
            return

        chunk = indata.copy()
        self.frames.append(chunk)

        try:
            self.SpeechToTextClient.send_audio_frame(chunk.tobytes())
        except Exception as e:
            print(f"Error sending audio frame: {e}")
            self.is_recording = False
            raise sd.CallbackStop

    def record_and_transcribe(self, duration=5):
        """Record for specified duration and transcribe(这个只有可能是被api调用了,暂时用不上)"""
        if self.start_recording():
            time.sleep(duration)
            return self.stop_recording()
        return None

    def _speak_blocking(self, text: str) -> None:
        """阻塞式的语音播放逻辑，供线程池调用。"""
        voice = self.SpeakerClient.call(
            model="qwen3-tts-flash",
            api_key=self.dashscope_api_key,
            text=text,
            voice="Cherry",
            language_type="Chinese",
            stream=True,
        )
        try:
            with sd.OutputStream(samplerate=24000, channels=1, dtype="int16") as stream:
                print("音频流已开启，准备播放...")
                for chunk in voice:
                    audio = chunk.output.audio
                    if audio.data is not None:
                        wav_bytes = base64.b64decode(audio.data)
                        audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
                        stream.write(audio_np)
                    if chunk.output.finish_reason == "stop":
                        print(
                            f"服务端标记结束，到期时间: {chunk.output.audio.expires_at}"
                        )
                print("音频数据接收完毕，等待播放完成...")
                time.sleep(0.8)
        except Exception as e:
            print(f"播放时发生错误: {e}")

        print("播放结束，音频流已自动关闭。")

    async def speak(self, text: str) -> None:
        """异步地播放语音，避免阻塞事件循环。"""
        # asyncio.to_thread 适合用来包裹“原本是同步/阻塞”的逻辑
        await asyncio.to_thread(self._speak_blocking, text)

    def cleanup(self):
        """Clean up audio resources"""
        if self.is_recording:
            self.stop_recording()
        if self._recognition_started:
            try:
                self.SpeechToTextClient.stop()
            except Exception as e:
                print(f"Error stopping recognition during cleanup: {e}")
            self._recognition_started = False
        self.frames.clear()

    @property
    def text(self) -> Optional[str]:
        if not self.transcribed_text:
            return None
        combined_text = "".join(self.transcribed_text)
        return combined_text

    @property
    def emotion(self) -> Optional[str]:
        return self.emo_tag

    def clear_text(self):
        self.transcribed_text.clear()


if __name__ == "__main__":
    print("开始测试SpeechService类...")

    # 创建SpeechService实例
    service = SpeechService()

    try:
        # 测试语音识别功能
        print("\n测试语音识别功能...")
        print("请说话,系统将持续识别您的语音")
        # 开始录音
        while True:
            if service.start_recording():
                while service.is_recording:
                    # 每隔1秒检查一次识别结果
                    time.sleep(0.01)
                    if service.text:
                        # 获取识别的文本
                        transcribed_text = service.text
                        print(f"识别结果: {transcribed_text}")
                        service.clear_text()
                        # 测试语音合成功能
                        if transcribed_text:
                            print("\n测试语音合成功能...")
                            print("将播放刚才识别的文本...")
                            asyncio.run(service.speak(transcribed_text))
                            service.stop_recording()
            else:
                transcribed_text = None
                print("录音启动失败")

            print("\n测试完成！")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    except KeyboardInterrupt as e:
        service.stop_recording()
    finally:
        # 清理资源
        service.stop_recording()
        service.cleanup()
