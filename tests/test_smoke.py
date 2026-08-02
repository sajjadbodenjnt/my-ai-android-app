def test_import_modules():
    import architecture, medical_rag, agent_pipeline
    assert hasattr(architecture, "SmallCausalTransformer")
    assert hasattr(medical_rag, "Retriever")
    assert hasattr(agent_pipeline, "MultiAgentSystem")
