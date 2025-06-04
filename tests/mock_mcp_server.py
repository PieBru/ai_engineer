from flask import Flask, Response, stream_with_context
import time
import json

app = Flask(__name__)

@app.route('/local-stream')
def local_stream():
    def generate_data():
        for i in range(5):
            yield f"Data chunk {i+1} from local stream at {time.strftime('%H:%M:%S')}\n"
            time.sleep(0.5)
        yield "End of local stream.\n"
    return Response(stream_with_context(generate_data()), mimetype='text/plain')

@app.route('/remote-sse')
def remote_sse():
    def generate_events():
        count = 0
        while count < 15: # Send more than the default max_events to test limit
            count += 1
            event_data = {"id": count, "message": f"SSE event number {count}", "timestamp": time.time()}
            yield f"data: {json.dumps(event_data)}\n\n"
            time.sleep(0.7)
            if count == 3: # Simulate an event with a specific name
                custom_event = {"status": "progress", "value": 50}
                yield f"event: custom_update\ndata: {json.dumps(custom_event)}\n\n"
    return Response(stream_with_context(generate_events()), mimetype='text/event-stream')

if __name__ == '__main__':
    # Make sure to install Flask: pip install Flask
    print("Mock MCP Server running on http://localhost:1957")
    print("Endpoints:")
    print("  Local Stream: http://localhost:1957/local-stream")
    print("  Remote SSE:   http://localhost:1957/remote-sse")
    app.run(debug=True, port=1957)
