#!/usr/bin/env python3
"""
Mock Inspection History for IronSight Command Center Semantic Search Demo.

This module generates realistic mock inspection history data for demonstrating
the Semantic Search functionality without requiring actual inspection data.

Requirements: Demo preparation (Task 16.2)
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger(__name__)


@dataclass
class InspectionRecord:
    """Single inspection record for semantic search demo."""
    record_id: str
    wagon_id: str
    timestamp: datetime
    image_path: str
    
    # Detection results
    damage_type: Optional[str] = None
    damage_severity: str = "none"  # none, minor, moderate, severe
    damage_description: Optional[str] = None
    
    # Processing info
    blur_score: float = 0.0
    enhancement_applied: bool = False
    deblur_applied: bool = False
    
    # OCR results
    ocr_result: Optional[str] = None
    ocr_confidence: float = 0.0
    
    # Detection confidence
    detection_confidence: float = 0.0
    
    # Embedding (placeholder for actual embedding)
    embedding_generated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "record_id": self.record_id,
            "wagon_id": self.wagon_id,
            "timestamp": self.timestamp.isoformat(),
            "image_path": self.image_path,
            "damage_type": self.damage_type,
            "damage_severity": self.damage_severity,
            "damage_description": self.damage_description,
            "blur_score": self.blur_score,
            "enhancement_applied": self.enhancement_applied,
            "deblur_applied": self.deblur_applied,
            "ocr_result": self.ocr_result,
            "ocr_confidence": self.ocr_confidence,
            "detection_confidence": self.detection_confidence,
            "embedding_generated": self.embedding_generated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InspectionRecord":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class MockInspectionHistory:
    """
    Generates and manages mock inspection history for demo purposes.
    
    Provides:
    - Realistic inspection records with various damage types
    - Searchable history with metadata
    - Export/import functionality
    """
    
    # Wagon ID patterns
    WAGON_PREFIXES = ["ABC", "XYZ", "RW", "FR", "DE", "UK", "PL"]
    
    # Damage types and descriptions
    DAMAGE_CATALOG = {
        "rust": [
            "Surface rust on door panel",
            "Rust spots near wheel assembly",
            "Corrosion on undercarriage",
            "Rust damage on side panel",
            "Oxidation on metal surface",
        ],
        "dent": [
            "Minor dent on side panel",
            "Impact damage on door",
            "Dent near coupling mechanism",
            "Multiple small dents on roof",
            "Large dent on corner section",
        ],
        "scratch": [
            "Surface scratches on paint",
            "Deep scratch on side panel",
            "Scratches near identification plate",
            "Multiple scratches from debris",
        ],
        "hole": [
            "Small hole in floor panel",
            "Puncture damage on side",
            "Hole near ventilation area",
        ],
        "crack": [
            "Hairline crack in weld joint",
            "Crack in structural beam",
            "Surface crack on panel",
        ],
        "wear": [
            "General wear on wheel assembly",
            "Worn brake components",
            "Surface wear from friction",
        ],
        "paint_damage": [
            "Peeling paint on exterior",
            "Faded paint on sun-exposed side",
            "Paint chips near edges",
        ],
    }
    
    # Severity weights (for random selection)
    SEVERITY_WEIGHTS = {
        "none": 0.3,
        "minor": 0.35,
        "moderate": 0.25,
        "severe": 0.1,
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize mock inspection history.
        
        Args:
            output_dir: Directory for storing history data
        """
        self.output_dir = output_dir or Path(__file__).parent / "generated" / "semantic_search"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.records: List[InspectionRecord] = []
        self._record_index: Dict[str, InspectionRecord] = {}
        
        logger.info(f"MockInspectionHistory initialized, output: {self.output_dir}")
    
    def generate_wagon_id(self) -> str:
        """Generate a realistic wagon ID."""
        prefix = random.choice(self.WAGON_PREFIXES)
        number = random.randint(10000, 99999)
        return f"{prefix}-{number}"
    
    def generate_record(
        self,
        timestamp: Optional[datetime] = None,
        wagon_id: Optional[str] = None,
        force_damage: bool = False
    ) -> InspectionRecord:
        """
        Generate a single inspection record.
        
        Args:
            timestamp: Record timestamp (random if None)
            wagon_id: Wagon ID (generated if None)
            force_damage: Force damage to be present
            
        Returns:
            Generated InspectionRecord
        """
        # Generate timestamp within last 30 days if not provided
        if timestamp is None:
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
        
        # Generate wagon ID if not provided
        if wagon_id is None:
            wagon_id = self.generate_wagon_id()
        
        # Generate record ID
        record_id = f"INS-{timestamp.strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"
        
        # Determine damage
        damage_type = None
        damage_severity = "none"
        damage_description = None
        
        # Select severity based on weights
        severity_roll = random.random()
        cumulative = 0
        for severity, weight in self.SEVERITY_WEIGHTS.items():
            cumulative += weight
            if severity_roll < cumulative:
                damage_severity = severity
                break
        
        if force_damage or damage_severity != "none":
            damage_type = random.choice(list(self.DAMAGE_CATALOG.keys()))
            damage_description = random.choice(self.DAMAGE_CATALOG[damage_type])
            if damage_severity == "none":
                damage_severity = "minor"
        
        # Generate processing info
        blur_score = random.uniform(0, 100)
        enhancement_applied = blur_score < 30  # Low light
        deblur_applied = blur_score > 50  # Blurry
        
        # Generate OCR results
        ocr_confidence = random.uniform(0.7, 0.99) if deblur_applied else random.uniform(0.85, 0.99)
        
        # Generate detection confidence
        detection_confidence = random.uniform(0.75, 0.98)
        
        # Create image path (placeholder)
        image_filename = f"inspection_{record_id}.jpg"
        image_path = str(self.output_dir / image_filename)
        
        record = InspectionRecord(
            record_id=record_id,
            wagon_id=wagon_id,
            timestamp=timestamp,
            image_path=image_path,
            damage_type=damage_type,
            damage_severity=damage_severity,
            damage_description=damage_description,
            blur_score=blur_score,
            enhancement_applied=enhancement_applied,
            deblur_applied=deblur_applied,
            ocr_result=wagon_id,  # OCR reads wagon ID
            ocr_confidence=ocr_confidence,
            detection_confidence=detection_confidence,
            embedding_generated=True,
        )
        
        return record
    
    def generate_history(
        self,
        num_records: int = 100,
        damage_ratio: float = 0.4
    ) -> List[InspectionRecord]:
        """
        Generate mock inspection history.
        
        Args:
            num_records: Number of records to generate
            damage_ratio: Ratio of records with damage (0-1)
            
        Returns:
            List of generated InspectionRecords
        """
        self.records = []
        self._record_index = {}
        
        # Generate records
        for i in range(num_records):
            force_damage = random.random() < damage_ratio
            record = self.generate_record(force_damage=force_damage)
            
            self.records.append(record)
            self._record_index[record.record_id] = record
        
        # Sort by timestamp (newest first)
        self.records.sort(key=lambda r: r.timestamp, reverse=True)
        
        logger.info(f"Generated {len(self.records)} inspection records")
        return self.records
    
    def search(
        self,
        query: str,
        limit: int = 20,
        min_confidence: float = 0.0
    ) -> List[InspectionRecord]:
        """
        Search inspection history (simple keyword matching for demo).
        
        Args:
            query: Search query
            limit: Maximum results
            min_confidence: Minimum detection confidence
            
        Returns:
            Matching records
        """
        query_lower = query.lower()
        results = []
        
        for record in self.records:
            if record.detection_confidence < min_confidence:
                continue
            
            # Check for matches
            match_score = 0
            
            # Check damage type
            if record.damage_type and record.damage_type.lower() in query_lower:
                match_score += 2
            
            # Check damage description
            if record.damage_description:
                desc_lower = record.damage_description.lower()
                for word in query_lower.split():
                    if word in desc_lower:
                        match_score += 1
            
            # Check wagon ID
            if record.wagon_id.lower() in query_lower:
                match_score += 3
            
            # Check severity
            if record.damage_severity in query_lower:
                match_score += 1
            
            # Generic damage query
            if "damage" in query_lower and record.damage_type:
                match_score += 1
            
            if match_score > 0:
                results.append((match_score, record))
        
        # Sort by match score
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [r[1] for r in results[:limit]]
    
    def get_damage_summary(self) -> Dict[str, int]:
        """Get summary of damage types in history."""
        summary = {}
        for record in self.records:
            if record.damage_type:
                summary[record.damage_type] = summary.get(record.damage_type, 0) + 1
        return summary
    
    def get_recent_records(
        self,
        hours: int = 24,
        limit: int = 50
    ) -> List[InspectionRecord]:
        """Get recent inspection records."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [r for r in self.records if r.timestamp >= cutoff]
        return recent[:limit]
    
    def export_to_json(self, filepath: Optional[Path] = None) -> Path:
        """
        Export history to JSON file.
        
        Args:
            filepath: Output file path
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = self.output_dir / "inspection_history.json"
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "num_records": len(self.records),
            "records": [r.to_dict() for r in self.records]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.records)} records to {filepath}")
        return filepath
    
    def import_from_json(self, filepath: Path) -> int:
        """
        Import history from JSON file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Number of records imported
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.records = [InspectionRecord.from_dict(r) for r in data["records"]]
        self._record_index = {r.record_id: r for r in self.records}
        
        logger.info(f"Imported {len(self.records)} records from {filepath}")
        return len(self.records)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the inspection history."""
        if not self.records:
            return {"num_records": 0}
        
        damage_count = sum(1 for r in self.records if r.damage_type)
        enhanced_count = sum(1 for r in self.records if r.enhancement_applied)
        deblurred_count = sum(1 for r in self.records if r.deblur_applied)
        
        return {
            "num_records": len(self.records),
            "damage_count": damage_count,
            "damage_ratio": damage_count / len(self.records),
            "enhanced_count": enhanced_count,
            "deblurred_count": deblurred_count,
            "damage_summary": self.get_damage_summary(),
            "date_range": {
                "oldest": min(r.timestamp for r in self.records).isoformat(),
                "newest": max(r.timestamp for r in self.records).isoformat(),
            }
        }


def create_mock_inspection_history(
    output_dir: Optional[Path] = None,
    num_records: int = 100
) -> MockInspectionHistory:
    """
    Factory function to create and populate MockInspectionHistory.
    
    Args:
        output_dir: Output directory
        num_records: Number of records to generate
        
    Returns:
        Populated MockInspectionHistory instance
    """
    history = MockInspectionHistory(output_dir=output_dir)
    history.generate_history(num_records=num_records)
    return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("📋 Mock Inspection History Generator")
    print("=" * 60)
    
    # Create and populate history
    history = create_mock_inspection_history(num_records=100)
    
    # Print stats
    stats = history.get_stats()
    print(f"\nGenerated {stats['num_records']} records")
    print(f"Damage ratio: {stats['damage_ratio']:.1%}")
    print(f"Enhanced: {stats['enhanced_count']}")
    print(f"Deblurred: {stats['deblurred_count']}")
    print(f"\nDamage summary:")
    for damage_type, count in stats['damage_summary'].items():
        print(f"  {damage_type}: {count}")
    
    # Test search
    print("\n🔍 Testing search...")
    results = history.search("rust damage", limit=5)
    print(f"Found {len(results)} results for 'rust damage'")
    for r in results[:3]:
        print(f"  - {r.wagon_id}: {r.damage_description}")
    
    # Export
    export_path = history.export_to_json()
    print(f"\n📁 Exported to: {export_path}")
    
    print("\n✅ Mock inspection history generation complete!")
