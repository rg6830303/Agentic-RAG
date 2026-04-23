-- Physics derivation checkpoints stored as a simple reference table
CREATE TABLE derivation_notes (
    topic TEXT,
    principle TEXT,
    result TEXT
);

INSERT INTO derivation_notes(topic, principle, result) VALUES
('Electrostatics', 'Coulomb law', 'Force varies inversely with square of separation distance'),
('Current Electricity', 'Microscopic current model', 'I equals number density times charge times area times drift speed'),
('Ray Optics', 'Lens formula', 'One over focal length equals one over image distance minus one over object distance');
