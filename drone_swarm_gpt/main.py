from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import timm
import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModel


class EncodedFeatures(NamedTuple):
    """Container for encoded features from different modalities."""

    vision: Optional[torch.Tensor]
    video: Optional[torch.Tensor]
    text: Optional[torch.Tensor]
    location: Optional[torch.Tensor]


@dataclass
class DroneState:
    """Represents the current state of a drone."""

    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    battery: float
    status: str


class MultiModalEncoder(nn.Module):
    """Encodes different input modalities into a common embedding space."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        logger.info(
            f"Initializing MultiModalEncoder with feature_dim={feature_dim}"
        )

        # Vision encoder with explicit shape handling
        base_model = timm.create_model(
            "resnet18", pretrained=True, features_only=True
        )
        self.vision_backbone = base_model
        backbone_channels = base_model.feature_info.channels()[
            -1
        ]  # Get channels from last layer

        self.vision_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_channels, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # Video encoder with matching dimensions
        self.video_encoder = nn.Sequential(
            nn.Conv3d(
                3,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.Conv3d(
                64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(
            "bert-base-uncased"
        )
        self.text_projection = nn.Sequential(
            nn.Linear(768, feature_dim), nn.LayerNorm(feature_dim)
        )

        # Location encoder
        self.location_encoder = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        logger.info("MultiModalEncoder initialization complete")

    def forward(
        self,
        vision_input: Optional[torch.Tensor] = None,
        video_input: Optional[torch.Tensor] = None,
        text_input: Optional[torch.Tensor] = None,
        location_input: Optional[torch.Tensor] = None,
    ) -> EncodedFeatures:
        """Encode each modality separately with proper shape handling."""
        vision_features = None
        video_features = None
        text_features = None
        location_features = None

        try:
            if vision_input is not None:
                logger.debug(
                    f"Vision input shape: {vision_input.shape}"
                )
                # Get features from all stages
                features = self.vision_backbone(vision_input)
                # Use the last feature map
                x = features[-1]
                logger.debug(
                    f"Vision backbone output shape: {x.shape}"
                )
                vision_features = self.vision_projection(x)
                logger.debug(
                    f"Vision projection output shape: {vision_features.shape}"
                )

            if video_input is not None:
                logger.debug(
                    f"Video input shape: {video_input.shape}"
                )
                video_features = self.video_encoder(video_input)
                logger.debug(
                    f"Video features shape: {video_features.shape}"
                )

            if text_input is not None:
                logger.debug(f"Text input shape: {text_input.shape}")
                text_output = self.text_encoder(text_input)[0][:, 0]
                text_features = self.text_projection(text_output)
                logger.debug(
                    f"Text features shape: {text_features.shape}"
                )

            if location_input is not None:
                logger.debug(
                    f"Location input shape: {location_input.shape}"
                )
                location_features = self.location_encoder(
                    location_input
                )
                logger.debug(
                    f"Location features shape: {location_features.shape}"
                )

        except Exception as e:
            logger.error(f"Error in encoder forward pass: {str(e)}")
            raise

        return EncodedFeatures(
            vision_features,
            video_features,
            text_features,
            location_features,
        )


class DroneTransformer(nn.Module):
    """Transformer model for processing encoded features."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
    ):
        super().__init__()
        logger.info(
            f"Initializing DroneTransformer with feature_dim={feature_dim}"
        )

        self.feature_dim = feature_dim

        # Modality type embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(4, feature_dim)
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        logger.info("DroneTransformer initialization complete")

    def forward(self, features: EncodedFeatures) -> torch.Tensor:
        """Process encoded features through transformer with shape logging."""
        try:
            # Create feature tensor with padding for missing modalities
            batch_size = next(
                f.shape[0] for f in features if f is not None
            )
            feature_tensors = []

            for idx, feat in enumerate(features):
                if feat is not None:
                    logger.debug(
                        f"Modality {idx} shape: {feat.shape}"
                    )
                    feature_tensors.append(
                        feat + self.modality_embeddings[idx]
                    )
                else:
                    padding = torch.zeros(
                        batch_size,
                        self.feature_dim,
                        device=self.modality_embeddings.device,
                    )
                    feature_tensors.append(
                        padding + self.modality_embeddings[idx]
                    )

            # Stack features
            x = torch.stack(
                feature_tensors, dim=1
            )  # [batch_size, 4, feature_dim]
            logger.debug(f"Combined features shape: {x.shape}")

            # Transform
            transformed = self.transformer(x)
            logger.debug(
                f"Transformer output shape: {transformed.shape}"
            )

            # Project to output space
            output = self.output_projection(
                transformed.reshape(batch_size, -1)
            )
            logger.debug(f"Final output shape: {output.shape}")

            return output

        except Exception as e:
            logger.error(
                f"Error in transformer forward pass: {str(e)}"
            )
            raise


class DroneController(nn.Module):
    """
    Enhanced drone controller with optional formation control capabilities.
    Provides both individual drone control and coordinated swarm movements.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_drones: int = 3,
        enable_formations: bool = True,
    ):
        super().__init__()
        self.num_drones = num_drones
        self.feature_dim = feature_dim
        self.enable_formations = enable_formations

        logger.info(
            f"Initializing DroneController with {num_drones} drones"
        )
        logger.info(f"Formation control enabled: {enable_formations}")

        # Core action types
        self.action_types = {
            0: "HOVER",  # Maintain position
            1: "NAVIGATE",  # Move to target
            2: "SURVEY",  # Perform area survey
            3: "FOLLOW",  # Follow moving target
            4: "RTB",  # Return to base
        }

        # Safety constraints
        self.safety_limits = {
            "max_altitude_change": 3.0,  # meters
            "max_position_change": 0.2,  # degrees
            "max_velocity": 2.0,  # m/s
            "min_confidence": 0.4,  # minimum confidence threshold
        }

        # Core components
        self.context_processor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        self.action_components = nn.ModuleDict(
            {
                "position": nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.GELU(),
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, num_drones * 3),
                ),
                "velocity": nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.GELU(),
                    nn.Linear(feature_dim, num_drones),
                ),
                "action_type": nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.GELU(),
                    nn.Linear(
                        feature_dim,
                        num_drones * len(self.action_types),
                    ),
                ),
                "confidence": nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.GELU(),
                    nn.Linear(feature_dim // 2, num_drones),
                    nn.Sigmoid(),
                ),
            }
        )

        # Optional formation control components
        if enable_formations:
            self.formation_patterns = {
                "triangle": torch.tensor(
                    [[0.0, 0.0], [0.1, 0.17], [-0.1, 0.17]]
                ),
                "line": torch.tensor(
                    [[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]]
                ),
                "circle": torch.tensor(
                    [
                        [
                            0.15
                            * torch.cos(
                                torch.tensor([2 * torch.pi / 3 * i])
                            ),
                            0.15
                            * torch.sin(
                                torch.tensor([2 * torch.pi / 3 * i])
                            ),
                        ]
                        for i in range(3)
                    ]
                ),
            }

            self.formation_controller = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, len(self.formation_patterns)),
            )

            logger.info("Initialized formation control components")

        self.drone_states: Dict[int, DroneState] = {}
        logger.info("DroneController initialization complete")

    def _apply_safety_constraints(
        self, actions: torch.Tensor
    ) -> torch.Tensor:
        """Apply safety constraints to drone actions."""
        actions[:, :, 0:2] = torch.clamp(
            actions[:, :, 0:2],
            -self.safety_limits["max_position_change"],
            self.safety_limits["max_position_change"],
        )

        actions[:, :, 2] = torch.clamp(
            actions[:, :, 2],
            -self.safety_limits["max_altitude_change"],
            self.safety_limits["max_altitude_change"],
        )

        actions[:, :, 3] = torch.clamp(
            actions[:, :, 3],
            -self.safety_limits["max_velocity"],
            self.safety_limits["max_velocity"],
        )

        return actions

    def _select_formation(
        self, features: torch.Tensor
    ) -> Optional[str]:
        """Select formation pattern if formations are enabled."""
        if not self.enable_formations:
            return None

        formation_logits = self.formation_controller(features)
        formation_idx = torch.argmax(formation_logits, dim=1)
        formations = list(self.formation_patterns.keys())
        return formations[formation_idx[0]]

    def _apply_formation(
        self, actions: torch.Tensor, formation: Optional[str]
    ) -> torch.Tensor:
        """Apply formation adjustments if enabled."""
        if not formation or not self.enable_formations:
            return actions

        formation_pattern = self.formation_patterns[formation]
        for i in range(self.num_drones):
            actions[:, i, :2] += formation_pattern[i]

        return actions

    def forward(
        self, features: torch.Tensor, maintain_formation: bool = False
    ) -> List[torch.Tensor]:
        """
        Generate coordinated drone actions with optional formation maintenance.

        Args:
            features: Input features tensor
            maintain_formation: Whether to maintain formation (only used if formations enabled)
        """
        batch_size = features.size(0)
        context = self.context_processor(features)

        # Generate base actions
        positions = self.action_components["position"](context).view(
            batch_size, self.num_drones, 3
        )
        velocities = self.action_components["velocity"](context).view(
            batch_size, self.num_drones, 1
        )
        action_logits = self.action_components["action_type"](
            context
        ).view(batch_size, self.num_drones, len(self.action_types))
        confidences = self.action_components["confidence"](context)

        action_types = torch.argmax(action_logits, dim=2).unsqueeze(
            -1
        )

        # Combine components
        actions = torch.cat(
            [
                positions,
                velocities,
                action_types.float(),
                confidences.unsqueeze(-1),
            ],
            dim=2,
        )

        # Apply formation if enabled and requested
        if self.enable_formations and maintain_formation:
            formation = self._select_formation(features)
            actions = self._apply_formation(actions, formation)

        # Apply safety constraints
        actions = self._apply_safety_constraints(actions)

        return [actions[:, i, :] for i in range(self.num_drones)]


class DroneSwarmSystem(nn.Module):
    """Complete drone swarm control system."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_drones: int = 3,
        num_transformer_layers: int = 6,
    ):
        super().__init__()
        self.encoder = MultiModalEncoder(feature_dim)
        self.transformer = DroneTransformer(
            feature_dim, num_layers=num_transformer_layers
        )
        self.controller = DroneController(feature_dim, num_drones)

    def forward(self, *args, **kwargs) -> List[torch.Tensor]:
        """Process inputs through entire pipeline."""
        encoded_features = self.encoder(*args, **kwargs)
        transformed_features = self.transformer(encoded_features)
        drone_actions = self.controller(transformed_features)
        return drone_actions
