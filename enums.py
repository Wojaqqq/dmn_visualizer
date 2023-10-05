from enum import Enum


class ElementsType(Enum):
    NODE_INPUT_DATA = 'node_input_data'
    TEXT = 'text'
    NODE_DECISION = 'node_decision'
    ARROW = 'arrow'
    NODE_KNOWLEDGE_SOURCE = 'node_knowledge_source'
    NODE_KNOWLEDGE_MODEL = 'node_knowledge_model'
