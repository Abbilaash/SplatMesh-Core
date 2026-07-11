package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

const (
	Port         = ":3000"
	BaseDataDir  = "../data"
	MaxQueueSize = 10
	WorkerCount  = 4
)

type Frame struct {
	Data      []byte
	Timestamp int64
	Index     int
	Pitch     float32
	Roll      float32
	Yaw       float32
	Dir       string // Tells the worker which folder to save to
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
	if err := os.MkdirAll(BaseDataDir, 0755); err != nil {
		log.Fatalf("Directory creation failed: %v", err)
	}

	frameChannel := make(chan Frame, MaxQueueSize)
	for i := 0; i < WorkerCount; i++ {
		go diskWorker(i, frameChannel)
	}

	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		handleConnection(w, r, frameChannel)
	})
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

	var currentSessionDir string

	for {
		messageType, reader, err := conn.NextReader()
		if err != nil {
			break
		}

		// Handle Control Commands (Start/Stop)
		if messageType == websocket.TextMessage {
			cmdBuf := make([]byte, 16)
			n, _ := reader.Read(cmdBuf)
			cmd := string(cmdBuf[:n])

			if cmd == "START" {
				sessionName := fmt.Sprintf("session_%d", time.Now().Unix())
				currentSessionDir = filepath.Join(BaseDataDir, sessionName)
				os.MkdirAll(currentSessionDir, 0755)
				log.Printf("[GO RELAY] Started new session: %s", sessionName)
			} else if cmd == "STOP" {
				log.Printf("[GO RELAY] Stopped session.")
				currentSessionDir = ""
			}
			continue
		}

		// Handle Binary Image Frames
		if messageType != websocket.BinaryMessage || currentSessionDir == "" {
			continue
		}

		// 1. Read the 32-byte Header
		headerBuf := make([]byte, 32)
		if _, err := io.ReadFull(reader, headerBuf); err != nil {
			continue
		}

		ts := int64(binary.LittleEndian.Uint64(headerBuf[0:8]))
		pitch := math.Float32frombits(binary.LittleEndian.Uint32(headerBuf[8:12]))
		roll := math.Float32frombits(binary.LittleEndian.Uint32(headerBuf[12:16]))
		yaw := math.Float32frombits(binary.LittleEndian.Uint32(headerBuf[16:20]))

		// 2. Read the remaining JPEG data
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
		f := Frame{
			Data: buf, Timestamp: ts, Index: frameIndex,
			Pitch: pitch, Roll: roll, Yaw: yaw, Dir: currentSessionDir,
		}
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
		filename := filepath.Join(f.Dir, fmt.Sprintf("frame_%d_%06d_P%.1f_R%.1f_Y%.1f.jpg",
			f.Timestamp, f.Index, f.Pitch, f.Roll, f.Yaw))
		_ = os.WriteFile(filename, f.Data, 0644)
		framePool.Put(f.Data)
	}
}
