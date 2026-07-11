# SplatMesh Capture — Quick Setup

### 1. Run the Server (Native Windows only)

Do **not** run in WSL. Open PowerShell in the `go_relay` directory:

```powershell
go build -o relay.exe main.go
.\relay.exe

```

*(Note: The server will automatically create a `../data/` directory one level up to store your scans).*

### 2. Browser Config (The "Must-Haves")

Mobile browsers block WebSockets and device sensors on insecure local IPs by default.

* **Chrome Flag:** Go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure` on your phone.
* **Add:** `[http://10.91.53.25:3000](http://10.91.53.25:3000)`
* **Restart:** Relaunch Chrome.

### 3. Firewall (Run as Admin)

If the phone can't connect, run this in an elevated PowerShell:

```powershell
netsh advfirewall firewall add rule name="SplatMeshGoRelay" dir=in action=allow protocol=TCP localport=3000

```

---

### 4. How to Scan (New UI)

1. **Unlock Sensors:** When you open the page, you **must** tap the blue `Enable Camera & Sensors` button. This gives the browser permission to read the hardware IMU (Gyroscope/Accelerometer).
2. **Record:** Tap the red shutter button at the bottom. The 3x3 alignment grid will appear, and a red pulsing dot will confirm data is streaming.
3. **Sessions:** Every time you start and stop recording, the Go server automatically groups those frames into a new, isolated folder (e.g., `data/session_1720700000/`).

### 5. Data Format (For the ML / Python Team)

The phone streams a 32-byte binary header followed by a 640px JPEG. The Go server unpacks this and saves the IMU data directly into the filenames.

Inside the session folder, files look like this:
`frame_1720700000_000001_P12.5_R-2.1_Y90.0.jpg`

* **`1720700000`**: Unix Timestamp (ms)
* **`000001`**: Frame Index
* **`P12.5`**: Pitch (Beta/X-axis tilt in degrees)
* **`R-2.1`**: Roll (Gamma/Y-axis tilt in degrees)
* **`Y90.0`**: Yaw (Alpha/Z-axis compass heading in degrees)

---

### 6. Troubleshooting

* **"Bind Forbidden/Access Denied":** A process is locking the port.

```powershell
taskkill /F /IM relay.exe /T

```
