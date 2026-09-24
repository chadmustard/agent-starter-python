import logging

from dotenv import load_dotenv
from livekit.agents import (
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics

from caddie import CaddieData, RoundSetupAgent
from golf_api import OpenGolfAPI
from publisher import ScorecardPublisher

logger = logging.getLogger("agent")

load_dotenv(".env.local")

server = AgentServer()


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    publisher = ScorecardPublisher(ctx.room)
    golf_api = OpenGolfAPI()
    ctx.add_shutdown_callback(golf_api.aclose)

    # Set up a voice AI pipeline using AssemblyAI, Fish Audio, and the LiveKit turn detector
    session = AgentSession[CaddieData](
        userdata=CaddieData(golf_api=golf_api, publish=publisher.publish),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-3-5-pro", language="en"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"
        ),
        turn_handling=TurnHandlingOptions(
            # The LiveKit turn detector determines when the user is done speaking and the agent should respond.
            # TurnDetector is an end-of-turn model that listens to the user's audio directly, combining
            # semantic understanding with acoustic cues (intonation, pitch, rhythm) for state-of-the-art accuracy.
            # AgentSession supplies the required VAD automatically.
            # See more at https://docs.livekit.io/agents/build/turns
            turn_detection=inference.TurnDetector(),
            # Adaptive interruptions use the turn detector to tell a real interruption from a
            # backchannel like "mhm" or "right", so the agent keeps talking through the latter.
            interruption={"mode": "adaptive"},
            # allow the LLM to generate a response while waiting for the end of turn
            # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
            preemptive_generation={"enabled": True},
        ),
        # Expressive mode injects the TTS provider's markup guide into the LLM prompt, so the model
        # emits inline delivery tags (emotion, pacing, non-verbal sounds) that the TTS renders and
        # the transcript never shows. Requires a TTS model that supports markup, such as the Fish
        # Audio model above.
        expressive=True,
    )

    # Join the room and connect to the user. Connecting before session.start()
    # makes the local participant available, so the scorecard RPC is
    # registered before the frontend could first call it.
    await ctx.connect()
    publisher.register_rpc()

    # Start the session, which initializes the voice pipeline and warms up the models.
    await session.start(
        agent=RoundSetupAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    # Push the initial (empty) scorecard so the frontend has something to
    # render right away, even before the golfer says anything.
    await session.userdata.push()

    # Play a "thinking" sound automatically while tool calls are in flight, to
    # make waits feel more natural.
    # See more at https://docs.livekit.io/agents/multimodality/audio/background-audio/
    background_audio = BackgroundAudioPlayer(
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
        ],
    )
    await background_audio.start(room=ctx.room, agent_session=session)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = anam.AvatarSession(
    #     persona_config=anam.PersonaConfig(
    #         name="...",
    #         avatarId="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/anam
    #     ),
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
