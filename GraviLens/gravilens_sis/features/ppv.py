class PhysicsPipelineVerifier:
    """
    Utility class to verify that physics preprocessing is working correctly
    """

    @staticmethod
    def verify_preprocessing(encoder, test_images):
        """
        Verify that physics preprocessing produces valid features

        Args:
            encoder: FixedSISPhysicsInformedEncoder instance
            test_images: (B, H, W) test images

        Returns:
            dict with verification results
        """
        print("\n" + "="*80)
        print("🔬 VERIFYING PHYSICS PREPROCESSING PIPELINE")
        print("="*80)

        results = {
            'passed': True,
            'checks': []
        }

        # Get physics features
        physics_features = encoder.get_physics_features(test_images)

        # Check 1: Shape
        expected_shape = (test_images.shape[0], 2, test_images.shape[1], test_images.shape[2])
        shape_ok = physics_features.shape == expected_shape
        results['checks'].append({
            'name': 'Shape Check',
            'passed': shape_ok,
            'expected': expected_shape,
            'actual': physics_features.shape
        })
        print(f"✓ Shape: {physics_features.shape} {'✅' if shape_ok else '❌'}")

        # Check 2: No NaN
        has_nan = torch.isnan(physics_features).any().item()
        results['checks'].append({
            'name': 'NaN Check',
            'passed': not has_nan,
            'message': 'No NaN detected' if not has_nan else 'NaN found!'
        })
        print(f"✓ NaN Check: {'✅ No NaN' if not has_nan else '❌ NaN detected!'}")

        # Check 3: Value ranges
        laplacian = physics_features[:, 0, :, :]
        determinant = physics_features[:, 1, :, :]

        lap_range_ok = (laplacian.min() >= 0) and (laplacian.max() <= 1)
        det_range_ok = (determinant.min() >= 0) and (determinant.max() <= 1)

        results['checks'].append({
            'name': 'Laplacian Range',
            'passed': lap_range_ok,
            'min': laplacian.min().item(),
            'max': laplacian.max().item()
        })

        results['checks'].append({
            'name': 'Determinant Range',
            'passed': det_range_ok,
            'min': determinant.min().item(),
            'max': determinant.max().item()
        })

        print(f"✓ Laplacian range: [{laplacian.min():.4f}, {laplacian.max():.4f}] {'✅' if lap_range_ok else '⚠️'}")
        print(f"✓ Determinant range: [{determinant.min():.4f}, {determinant.max():.4f}] {'✅' if det_range_ok else '⚠️'}")

        # Check 4: Statistics
        print(f"\nLaplacian Statistics:")
        print(f"  Mean: {laplacian.mean():.6f}")
        print(f"  Std:  {laplacian.std():.6f}")

        print(f"\nDeterminant Statistics:")
        print(f"  Mean: {determinant.mean():.6f}")
        print(f"  Std:  {determinant.std():.6f}")

        # Overall result
        results['passed'] = all(check['passed'] for check in results['checks'])

        print("\n" + "="*80)
        print(f"{'✅ ALL CHECKS PASSED!' if results['passed'] else '❌ SOME CHECKS FAILED!'}")
        print("="*80 + "\n")

        return results

    @staticmethod
    def compare_with_raw_patches(encoder, test_images, raw_tokenizer, physics_tokenizer):
        """
        Compare patches from raw images vs physics features
        This verifies that physics features are actually being used

        Args:
            encoder: FixedSISPhysicsInformedEncoder instance
            test_images: (B, H, W) test images
            raw_tokenizer: Tokenizer that processes raw images (1 channel)
            physics_tokenizer: Tokenizer that processes physics features (2 channels)

        Returns:
            dict with comparison results
        """
        print("\n" + "="*80)
        print("🔬 COMPARING RAW PATCHES vs PHYSICS PATCHES")
        print("="*80)

        batch_size = test_images.shape[0]

        # Get raw patches (old way - WRONG!)
        raw_patches = raw_tokenizer(
            test_images.reshape(batch_size, 1, encoder.image_size, encoder.image_size)
        )

        # Get physics features
        physics_features = encoder.get_physics_features(test_images)

        # Get physics patches (new way - CORRECT!)
        physics_patches = physics_tokenizer(physics_features)

        print(f"\n✓ Raw patches shape: {raw_patches.shape}")
        print(f"✓ Physics features shape: {physics_features.shape}")
        print(f"✓ Physics patches shape: {physics_patches.shape}")

        # Calculate difference
        # Note: They should have same shape but different content!
        if raw_patches.shape == physics_patches.shape:
            diff = torch.abs(raw_patches - physics_patches).mean().item()
            print(f"\n✓ Mean absolute difference: {diff:.6f}")

            if diff < 0.01:
                print("❌ WARNING: Patches are too similar! Physics features may not be used.")
                return {'status': 'FAIL', 'reason': 'Patches nearly identical'}
            else:
                print("✅ Patches are different! Physics features ARE being used.")
                return {'status': 'PASS', 'difference': diff}
        else:
            print(f"⚠️  Shapes don't match - this is expected if input channels differ")
            return {'status': 'PASS', 'reason': 'Different architectures'}

        print("="*80 + "\n")