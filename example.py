from loguru import logger
import torch
from transformers import AutoTokenizer
from drone_swarm_gpt.main import DroneSwarmSystem


def load_sample_data(
    batch_size: int = 2, image_size: int = 224, video_frames: int = 8
):
    """Create sample input data for testing."""
    try:
        # Sample data creation with fixed shapes
        image_data = torch.randn(
            batch_size, 3, image_size, image_size
        )
        video_data = torch.randn(
            batch_size, 3, video_frames, image_size, image_size
        )

        # Text tokenization
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        text_samples = [
            "drone move to coordinate 42.3601° N, 71.0589° W",
            "survey the area with thermal imaging",
        ]
        text_tokens = tokenizer(
            text_samples, padding=True, return_tensors="pt"
        )["input_ids"]

        # Location data
        location_data = torch.tensor(
            [
                [42.3601, -71.0589, 100.0],  # Boston
                [40.7128, -74.0060, 150.0],  # NYC
            ]
        )

        logger.info(
            f"Created sample data with batch size {batch_size}"
        )
        return image_data, video_data, text_tokens, location_data

    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}")
        raise


def main():
    """Main function demonstrating the drone swarm system."""
    try:
        # Initialize system
        system = DroneSwarmSystem(
            feature_dim=256, num_drones=3, num_transformer_layers=6
        )

        # Load sample data
        image_data, video_data, text_data, location_data = (
            load_sample_data(batch_size=2)
        )

        # Process through system
        logger.info("Processing inputs through drone swarm system")
        drone_actions = system(
            vision_input=image_data,
            video_input=video_data,
            text_input=text_data,
            location_input=location_data,
        )

        # Process results
        for drone_idx, actions in enumerate(drone_actions):
            logger.info(f"\nDrone {drone_idx} actions:")
            for batch_idx in range(actions.shape[0]):
                (
                    delta_lat,
                    delta_lon,
                    delta_alt,
                    velocity,
                    action_type,
                    confidence,
                ) = actions[batch_idx].tolist()
                logger.info(
                    f"Batch {batch_idx}:\n"
                    f"  Position adjustment: ({delta_lat:.2f}°, {delta_lon:.2f}°, {delta_alt:.2f}m)\n"
                    f"  Velocity: {velocity:.2f} m/s\n"
                    f"  Action type: {action_type:.0f}\n"
                    f"  Confidence: {confidence:.2%}"
                )

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        "drone_system.log",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(lambda msg: print(msg), level="INFO")

    main()
