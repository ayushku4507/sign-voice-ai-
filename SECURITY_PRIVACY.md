# Security and Privacy Notes

- Ask permission before accessing camera or microphone.
- Stop media tracks when the user leaves the feature or disables capture.
- Do not persist raw video/audio by default.
- If cloud inference is added, disclose exactly what is sent, where, retention period and deletion process.
- Use HTTPS in deployment.
- Keep secrets in server-side environment variables; never commit `.env` files.
- Restrict CORS to known production origins.
- Validate request payload size and type before accepting uploads.
- Add rate limiting and authentication before exposing costly inference endpoints.
- Provide clear controls to clear local history if history is added.
- Never claim an emergency service has been contacted unless a real integration confirms it.
