package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

const (
	Port         = ":3000"
	OutputDir    = "../data/raw_frames"
	MaxQueueSize = 10
	WorkerCount  = 4
)

type Frame struct {
	Data      []byte
	Timestamp int64
	Index     int
}

var (
	upgrader = websocket.Upgrader{
		ReadBufferSize:  1024 * 1024,
		WriteBufferSize: 1024,
		CheckOrigin:     func(r *http.Request) bool { return true },
	}
	framePool = sync.Pool{
		New: func() interface{} { return make([]byte, 0, 500*1024) },
	}
	frameIndex = 0
	indexLock  sync.Mutex
)

func main() {
	if err := os.MkdirAll(OutputDir, 0755); err != nil {
		log.Fatalf("Directory creation failed: %v", err)
	}

	frameChannel := make(chan Frame, MaxQueueSize)
	for i := 0; i < WorkerCount; i++ {
		go diskWorker(i, frameChannel)
	}

	// Route 1: WebSocket
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		handleConnection(w, r, frameChannel)
	})

	// Route 2: Serve HTML (Assuming phone_stream.html is in the folder one level up: ../mobile)
	// If phone_stream.html is in the same folder as this code, change "." to "."
	http.Handle("/", http.FileServer(http.Dir("../mobile")))

	fmt.Printf("[GO RELAY] Running on http://10.91.53.25%s\n", Port)
	if err := http.ListenAndServe(Port, nil); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

func handleConnection(w http.ResponseWriter, r *http.Request, ch chan<- Frame) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()
	log.Printf("[GO RELAY] Phone connected: %s", conn.RemoteAddr().String())

	for {
		messageType, reader, err := conn.NextReader()
		if err != nil {
			break
		}
		if messageType != websocket.BinaryMessage {
			continue
		}

		buf := framePool.Get().([]byte)[:0]
		tmp := make([]byte, 4096)
		for {
			n, err := reader.Read(tmp)
			if n > 0 {
				buf = append(buf, tmp[:n]...)
			}
			if err != nil {
				break
			}
		}

		indexLock.Lock()
		frameIndex++
		f := Frame{Data: buf, Timestamp: time.Now().UnixMilli(), Index: frameIndex}
		indexLock.Unlock()

		select {
		case ch <- f:
		default:
			framePool.Put(buf)
		}
	}
}

func diskWorker(id int, ch <-chan Frame) {
	for f := range ch {
		filename := filepath.Join(OutputDir, fmt.Sprintf("frame_%d_%06d.jpg", f.Timestamp, f.Index))
		_ = os.WriteFile(filename, f.Data, 0644)
		framePool.Put(f.Data)
	}
}
