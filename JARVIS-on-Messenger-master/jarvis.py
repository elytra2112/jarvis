import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import config
import modules

# Load access token and verify token from environment variables or config file
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN', config.ACCESS_TOKEN)
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN', config.VERIFY_TOKEN)

class WebhookHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/webhook'):
            query_params = dict(pair.split('=') for pair in self.path[1:].split('&'))
            if query_params.get('hub.verify_token') == VERIFY_TOKEN:
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(query_params.get('hub.challenge').encode())
            else:
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b'Error, wrong validation token')
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')

    def do_POST(self):
        if self.path.startswith('/webhook'):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)
            messaging_events = data.get('entry', [])[0].get('messaging', [])
            for event in messaging_events:
                sender_id = event['sender']['id']
                message = None
                
                # Handling messages
                if 'message' in event and 'text' in event['message']:
                    message_text = event['message']['text']
                    message = modules.search(message_text, sender=sender_id)
                
                # Handling postbacks
                if 'postback' in event and 'payload' in event['postback']:
                    postback_payload = event['postback']['payload']
                    message = modules.search(postback_payload, sender=sender_id, postback=True)
                
                # Send message if not None
                if message is not None:
                    payload = {
                        'recipient': {
                            'id': sender_id
                        },
                        'message': message
                    }
                    # You need to implement sending the response using requests library or other means
                    # Example: requests.post('https://graph.facebook.com/v2.6/me/messages', params={'access_token': ACCESS_TOKEN}, json=payload)
                    print(payload)  # Placeholder for sending the response
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')

def run_server(server_class=HTTPServer, handler_class=WebhookHandler, port=5000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
