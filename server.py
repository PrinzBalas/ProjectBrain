import asyncio
import websockets
import json

PORT = 8765


# -----------------------
# Placeholder AI function
# -----------------------
def ai_predict(state):
    """
    Replace this later with your NN inference.
    For now: echo demo
    """
    # state = [x, y, z, ...]
    # Example dummy logic:
    return [s * 0.5 for s in state]


# --------------------------------
# One connection = one agent
# --------------------------------
async def handle_client(websocket):
    print("Client connected")

    try:
        async for msg in websocket:
            data = json.loads(msg)

            # Expected format:
            # {"state": [0.5,1.0,2.2]}
            state = data.get("state", [])

            actions = ai_predict(state)

            response = {
                "actions": actions
            }

            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main():
    print(f"WebSocket server running on ws://localhost:{PORT}")
    async with websockets.serve(handle_client, "0.0.0.0", PORT):
        await asyncio.Future()  # forces server to run forever

if __name__ == "__main__":
    asyncio.run(main())


    