#!/usr/bin/env python3
"""Simple server with COOP/COEP headers for SharedArrayBuffer testing."""

import http.server
import socketserver

PORT = 8080

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server at http://localhost:{PORT}")
    print(f"Open http://localhost:{PORT}/pkg/ in browser")
    print("Then in console: await import('./js/bench.js').then(m => m.testChiSignalInfra(10000))")
    httpd.serve_forever()
