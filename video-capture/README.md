
# SplatMesh Capture — Quick Setup

### 1. Run the Server (Native Windows only)
Do **not** run in WSL. Open PowerShell in `go_relay` directory:
```powershell
go build -o relay.exe main.go
.\relay.exe

```

### 2. Browser Config (The "Must-Haves")

Mobile browsers block WebSockets on insecure local IPs by default.

* **Chrome Flag:** Go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure` on your phone.
* **Add:** `http://10.91.53.25:3000`
* **Restart:** Relaunch Chrome.

### 3. Firewall (Run as Admin)

If the phone can't connect:

```powershell
netsh advfirewall firewall add rule name="SplatMeshGoRelay" dir=in action=allow protocol=TCP localport=3000

```

### 4. Troubleshooting

* **"Bind Forbidden/Access Denied":** A process is locking the port.
```powershell
taskkill /F /IM relay.exe /T

```


* **Changes not showing:** Append `?v=X` to your URL (e.g., `http://10.91.53.25:3000/?v=2`) to force a cache bypass.
* **Video is zoomed:** Always use landscape orientation.
