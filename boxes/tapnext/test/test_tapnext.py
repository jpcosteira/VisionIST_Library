#!/usr/bin/env python3
"""Test script for TAPNext gRPC service - sequential frame processing."""

import sys
sys.path.append("../protos")

import grpc
import pipeline_pb2, pipeline_pb2_grpc, aux
import json
import cv2
import numpy as np

def load_video_frames(video_path):
    """Load frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def bytes_to_tensor(b: bytes):
    """Deserialize tensor from bytes."""
    import torch
    import io as bio
    return torch.load(bio.BytesIO(b), weights_only=False)

def process_response(response, frame_idx, orig_h, orig_w, frames, out):
    """Process a tracking response and draw on frame."""
    import matplotlib.pyplot as plt
    
    if not response.data or "tracks" not in response.data:
        print(f"  Frame {frame_idx}: No tracks data in response")
        return False
    
    # Check for observation matrix output (Tomasi-Kanade)
    if response.data and "observation_matrix" in response.data:
        obs_bytes = aux.unwrap_value(response.data["observation_matrix"])
        import torch
        import io as bio
        P_tensor = torch.load(bio.BytesIO(obs_bytes), weights_only=False)
        print(f"\n  Observation Matrix (P): {P_tensor.shape}")
    
    # Deserialize tracks and visibles
    tracks_tensor = bytes_to_tensor(aux.unwrap_value(response.data["tracks"]))
    visibles_tensor = bytes_to_tensor(aux.unwrap_value(response.data["visibles"]))
    
    print(f"\n  Tracks tensor: {tracks_tensor.shape}")
    print(f"  Visibles tensor: {visibles_tensor.shape}")
    
    # Convert to numpy
    tracks_np = tracks_tensor.numpy()  # Shape: (num_frames, num_points, 2)
    visibles_np = visibles_tensor.numpy()
    
    if len(tracks_np) == 0:
        print(f"  Frame {frame_idx}: No tracking results")
        return False
    
    # Use the last frame's results (since we send one frame at a time)
    frame_tracks = tracks_np[-1]  # [num_points, 2]
    frame_visibles = visibles_np[-1]  # [num_points]
    
    frame = frames[frame_idx].copy()
    
    # Draw each track point
    for pt_idx in range(len(frame_tracks)):
        if not frame_visibles[pt_idx]:
            continue
            
        coords = frame_tracks[pt_idx]
        y, x = float(coords[0]), float(coords[1])
        
        # Skip invalid coordinates (NaN or None)
        if np.isnan(x) or np.isnan(y):
            continue
        
        # Color based on track ID
        color = tuple(int(c) for c in np.array(plt.cm.rainbow(pt_idx / len(frame_tracks)))[:3] * 255)
        
        cv2.circle(frame, (int(x), int(y)), radius=4, color=color, thickness=-1)
    
    out.write(frame)
    print(f"  Frame {frame_idx}: OK ({len(frame_tracks)} points)")
    return True

def main():
    import matplotlib.pyplot as plt
    
    # Connect to the service
    channel = grpc.insecure_channel('localhost:8061')
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)
    
    print("Testing TAPNext point tracking service (sequential mode)...")
    
    # Load test video frames
    video_path = "./apple.mp4"
    try:
        frames = load_video_frames(video_path)
        print(f"Loaded {len(frames)} frames from test video")
        orig_h, orig_w = frames[0].shape[:2]
    except Exception as e:
        print(f"Failed to load video: {e}")
        return
    
    # Setup video writer
    output_video_path = "./output_tracking_sequential.mp4"
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_w, orig_h))
    
    print(f"\nProcessing {len(frames)} frames sequentially...")
    
    # Clear any previous tracking state with explicit reset
    print("Sending initial reset command...")
    reset_request = pipeline_pb2.Envelope(
        config_json=json.dumps({"tapnext": {"command": "reset"}})
    )
    stub.Process(reset_request)
    
    # Process each frame individually
    for frame_idx in range(len(frames)):
        # Encode single frame
        _, buf = cv2.imencode('.jpg', frames[frame_idx])
        frame_bytes_list = [buf.tobytes()]
        
        config = {
            "tapnext": {
                "command": "track",
                "parameters": {"grid_size": 30}
            },
            "stream": max(0, len(frames) - 1 - frame_idx)
        }
        
        request = pipeline_pb2.Envelope(
            config_json=json.dumps(config),
            data={"images": aux.wrap_value(frame_bytes_list)}
        )
        
        response = stub.Process(request)
        
        if not process_response(response, frame_idx, orig_h, orig_w, frames, out):
            break
    
    # Send reset command at the end
    print("\nSending reset command...")
    reset_request = pipeline_pb2.Envelope(
        config_json=json.dumps({
            "tapnext": {"command": "reset"}
        })
    )
    reset_response = stub.Process(reset_request)
    reset_result = json.loads(reset_response.config_json)
    print(f"Reset response: {reset_result}")
    
    out.release()
    print(f"\n✓ Output video saved to: {output_video_path}")
    print("✓ All tests completed!")

if __name__ == "__main__":
    main()
