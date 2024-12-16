import torch
import torch.nn as nn
import timm
from transformers import VideoMAEFeatureExtractor, VideoMAEModel, VideoMAEConfig, AutoTokenizer, AutoModel
import torch.nn.functional as F
from loguru import logger
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from torchvision import transforms

@dataclass
class DroneData:
    """Container for individual drone sensor and state data."""
    video: torch.Tensor          # Shape: (B, F, C, H, W)
    image: torch.Tensor          # Shape: (B, C, H, W)
    position: torch.Tensor       # Shape: (B, 3) - xyz coordinates
    orientation: torch.Tensor    # Shape: (B, 3) - roll, pitch, yaw
    velocity: torch.Tensor       # Shape: (B, 3) - velocity vector
    battery: torch.Tensor        # Shape: (B, 1) - battery percentage
    timestamp: torch.Tensor      # Shape: (B, 1) - timestamp of data

class MultiStreamProcessor(nn.Module):
    """
    Processes multiple input streams from drone swarms using state-of-the-art 
    pretrained models for video and sensor data analysis.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_drones: int = 3,
        video_frames: int = 16,  # VideoMAE expects 16 frames
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_drones = num_drones
        self.device = device
        self.video_frames = video_frames
        
        logger.info(f"Initializing MultiStreamProcessor for {num_drones} drones")
        
        # Initialize image processors (ViT Large)
        self.image_processors = nn.ModuleList([
            timm.create_model(
                'vit_large_patch16_384',
                pretrained=True,
                num_classes=0,
                global_pool='token'
            ) for _ in range(num_drones)
        ])
        
        # Image feature projection
        self.image_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_drones)
        ])
        
        # Initialize VideoMAE components
        self.video_feature_extractor = VideoMAEFeatureExtractor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        
        self.video_processors = nn.ModuleList([
            VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            for _ in range(num_drones)
        ])
        
        # Video feature projection
        self.video_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_drones)
        ])
        
        # Sensor data processors
        self.sensor_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(11, feature_dim // 2),
                nn.GELU(),
                nn.LayerNorm(feature_dim // 2),
                nn.Dropout(0.1),
                nn.Linear(feature_dim // 2, feature_dim),
                nn.LayerNorm(feature_dim)
            ) for _ in range(num_drones)
        ])
        
        # Modality fusion
        self.modality_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 3, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_drones)
        ])
        
        # Cross-attention for information sharing
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Global context integration
        self.global_fusion = nn.Sequential(
            nn.Linear(feature_dim * num_drones, feature_dim * 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.to(device)
        logger.info("MultiStreamProcessor initialization complete")
    
    def process_image(
        self,
        image: torch.Tensor,
        processor_idx: int
    ) -> torch.Tensor:
        """Process high-resolution image using ViT Large."""
        try:
            with torch.no_grad():
                image_features = self.image_processors[processor_idx](image)
            return self.image_projections[processor_idx](image_features)
        except Exception as e:
            logger.error(f"Error processing image stream {processor_idx}: {str(e)}")
            raise
    
    
    def process_video(
        self,
        video: torch.Tensor,
        processor_idx: int
    ) -> torch.Tensor:
        """
        Process video stream using VideoMAE with proper input type handling.
        
        Args:
            video: Input video tensor of shape (B, F, C, H, W)
            processor_idx: Index of the video processor to use
            
        Returns:
            torch.Tensor: Processed video features
        """
        try:
            B, F, C, H, W = video.shape
            logger.debug(f"Input video shape: {video.shape}")
            
            # Convert to numpy with proper format
            video = video.permute(0, 1, 3, 4, 2)  # B, F, H, W, C
            video_np = (video.cpu().numpy() * 255).astype(np.uint8)
            logger.debug(f"Converted video shape: {video_np.shape}, dtype: {video_np.dtype}")
            
            with torch.no_grad():
                processed_batches = []
                for batch_idx in range(B):
                    # Convert frames to list of numpy arrays
                    frame_list = [video_np[batch_idx, i] for i in range(F)]
                    
                    # Process through feature extractor
                    processed_video = self.video_feature_extractor(
                        frame_list,
                        return_tensors="pt",
                        do_rescale=True
                    )["pixel_values"].to(self.device)
                    
                    processed_batches.append(processed_video)
                
                # Combine processed batches
                processed_video = torch.cat(processed_batches, dim=0)
                logger.debug(f"Processed video shape: {processed_video.shape}")
                
                # Process through VideoMAE
                outputs = self.video_processors[processor_idx](
                    processed_video,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract features
                hidden_states = outputs.last_hidden_state
                pooled_features = hidden_states.mean(dim=1)
                projected_features = self.video_projections[processor_idx](pooled_features)
                
                logger.debug(f"Final features shape: {projected_features.shape}")
                return projected_features
                
        except Exception as e:
            logger.error(f"Error processing video stream {processor_idx}: {str(e)}")
            raise
    
    def process_sensor_data(
        self,
        drone_data: DroneData,
        processor_idx: int
    ) -> torch.Tensor:
        """Process sensor data."""
        try:
            sensor_vector = torch.cat([
                drone_data.position,
                drone_data.orientation,
                drone_data.velocity,
                drone_data.battery,
                drone_data.timestamp
            ], dim=1)
            return self.sensor_processors[processor_idx](sensor_vector)
        except Exception as e:
            logger.error(f"Error processing sensor data {processor_idx}: {str(e)}")
            raise
    
    def forward(
        self,
        drone_data_list: List[DroneData]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Process multi-modal data from all drones and generate shared understanding.
        """
        try:
            if len(drone_data_list) != self.num_drones:
                raise ValueError(f"Expected {self.num_drones} drones, got {len(drone_data_list)}")
            
            batch_size = drone_data_list[0].video.size(0)
            individual_features = []
            
            # Process each drone's data
            for i in range(self.num_drones):
                drone_data = drone_data_list[i]
                
                # Process each modality
                image_features = self.process_image(drone_data.image, i)
                video_features = self.process_video(drone_data.video, i)
                sensor_features = self.process_sensor_data(drone_data, i)
                
                # Combine modalities for this drone
                combined = torch.cat([
                    image_features,
                    video_features,
                    sensor_features
                ], dim=-1)
                
                # Fuse modalities
                fused_features = self.modality_fusion[i](combined)
                individual_features.append(fused_features)
            
            # Stack for cross-attention
            stacked_features = torch.stack(individual_features, dim=1)
            
            # Apply cross-attention for information sharing
            shared_features, _ = self.cross_attention(
                stacked_features,
                stacked_features,
                stacked_features
            )
            
            # Generate global context
            global_context = self.global_fusion(
                shared_features.reshape(batch_size, -1)
            )
            
            return global_context, [
                shared_features[:, i, :] for i in range(self.num_drones)
            ]
            
        except Exception as e:
            logger.error(f"Error in MultiStreamProcessor forward pass: {str(e)}")
            raise

def create_sample_data(
    batch_size: int = 2,
    num_drones: int = 3,
    video_frames: int = 16,
    video_size: int = 224,
    image_size: int = 384,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> List[DroneData]:
    """Create sample data with proper video format."""
    logger.info(f"Creating sample data for {num_drones} drones")
    
    drone_data_list = []
    
    for drone_idx in range(num_drones):
        try:
            # Create video tensor with values in [0, 1]
            video = torch.rand(batch_size, video_frames, 3, video_size, video_size)
            
            # Create image tensor
            image = torch.rand(batch_size, 3, image_size, image_size)
            
            # Other sensor data
            position = torch.randn(batch_size, 3)
            orientation = torch.randn(batch_size, 3)
            velocity = torch.randn(batch_size, 3)
            battery = torch.rand(batch_size, 1) * 100
            timestamp = torch.ones(batch_size, 1)
            
            drone_data = DroneData(
                video=video.to(device),
                image=image.to(device),
                position=position.to(device),
                orientation=orientation.to(device),
                velocity=velocity.to(device),
                battery=battery.to(device),
                timestamp=timestamp.to(device)
            )
            
            drone_data_list.append(drone_data)
            logger.debug(
                f"Created data for drone {drone_idx} - "
                f"Video shape: {video.shape}, "
                f"Video range: [{video.min():.3f}, {video.max():.3f}]"
            )
            
        except Exception as e:
            logger.error(f"Error creating sample data for drone {drone_idx}: {str(e)}")
            raise
    
    return drone_data_list

def main():
    """Demonstrate the usage of MultiStreamProcessor."""
    try:
        # Configuration
        batch_size = 2
        num_drones = 3
        feature_dim = 256
        video_frames = 16
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info("Initializing MultiStreamProcessor demonstration")
        
        # Initialize processor
        processor = MultiStreamProcessor(
            feature_dim=feature_dim,
            num_drones=num_drones,
            video_frames=video_frames,
            device=device
        )
        
        # Create sample data
        drone_data_list = create_sample_data(
            batch_size=batch_size,
            num_drones=num_drones,
            video_frames=video_frames
        )
        
        logger.info("Processing drone data through MultiStreamProcessor")
        
        # Forward pass
        global_context, individual_features = processor(drone_data_list)
        
        # Log results
        logger.info(f"Global context shape: {global_context.shape}")
        for i, features in enumerate(individual_features):
            logger.info(f"Drone {i} features shape: {features.shape}")
            logger.info(f"Feature statistics - Mean: {features.mean():.3f}, "
                       f"Std: {features.std():.3f}")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        "drone_system.log",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(lambda msg: print(msg), level="INFO")
    
    main()