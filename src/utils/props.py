import os

from typing import Dict

from log import info


def _find_models(root_dir, file_name):
    if not os.path.exists(root_dir):
        return {}

    ret = {}
    for username in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, username)): continue
        if os.path.exists(os.path.join(root_dir, username, file_name)):
            ret[username] = os.path.join(root_dir, username, file_name)
    return ret


class Props:
    """
    Global properties for the app
    """
    # Constants
    ELOQUENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

    PROD_SQL_DB_IP: str = "mysql.hybridcrowd.io"
    DEV_SQL_DB_IP: str = "dev.mysql.hybridcrowd.io"
    BURNER_SQL_DB_IP: str = "burner.mysql.hybridcrowd.io"
    CI_SQL_DB_IP = "mysql"

    PROD_SQL_DB_INSTANCE_NAME = "chat-api-1255:us-central1:eloquent-turk-sql"
    DEV_SQL_DB_INSTANCE_NAME = "chat-api-1255:us-central1:eloquent-turk-dev"
    BURNER_SQL_DB_INSTANCE_NAME = "chat-api-1255:us-central1:eloquent-turk-dev"
    CI_SQL_DB_INSTANCE_NAME = ""

    SQL_USER = "eloquent"
    SQL_AGENT_DB = "agent"

    PROD_CREDENTIALS_FILE = "credentials.json"
    DEV_CREDENTIALS_FILE = "credentials.dev.json"
    CI_CREDENTIALS_FILE = "credentials.ci.json"

    KNOL_TABLE = "knol"
    KNOL_HISTORY_TABLE = "knol_history"
    ELQ_PERMISSION_FOR_KNOL_VEIW = "elq_permission_for_knol_view"
    KNOL_TAG_INSTANCE = "knol_tag_instance"

    GRPC_PORT: int = 2109

    def __init__(self):
        # List all properties
        self.ELOQUENT_ROOT = Props.ELOQUENT_ROOT
        self.MODEL_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models")
        # Build
        self.BUILD_DATE: str = None
        self.IS_PROD: bool = None
        self.IS_DEPLOYED: bool = None
        self.IS_CI: bool = None
        # REST
        self.HTTP_PORT: int = None
        self.YAML_FILE: str = None

        # FAQ models
        self.MODEL_PATH_FAQ_NEURAL: Dict[str, str] = _find_models(os.path.join(self.MODEL_ROOT, "faq"), "faq_neural.pkl.gz")
        self.MODEL_PATH_FAQ_STATISTICAL: Dict[str, str] = _find_models(os.path.join(self.MODEL_ROOT, "faq"), "faq_statistical.pkl.gz")
        # Machine Reading models
        self.MODEL_PATH_MR: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../models/reader.pb.gz"
        )
        # IE Models
        self.MODEL_PATH_IE: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models/ie")
        # Intent Models
        self.MODEL_PATH_INTENT: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models/intent")
        # Emotion Models
        self.MODEL_PATH_EMOTION: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../models/emotion.pkl.gz"
        )
        # IDK Models
        self.MODEL_PATH_IDK: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../models/idk.pkl.gz"
        )
        # Enum IE Model
        self.MODEL_PATH_ENUM_IE: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../models/enum_ie.pkl.gz",
        )
        # SQL properties
        self.SQL_DB_IP: str = None
        self.SQL_DB_PORT: int = 3306
        self.SQL_DB_INSTANCE_NAME: str = None
        self.SQL_DB_USER: str = None
        self.SQL_DB_PASSWORD: str = None
        # GloVE
        self.GLOVE_PATH = (os.path.dirname(os.path.realpath(__file__)) + '/../../resources/glove.840B.300d.ser')
        self.BPE_PATH = (os.path.dirname(os.path.realpath(__file__)) + '/../../resources/bpe_encoder.pkl.gz')

    def local(self) -> 'Props':
        self.BUILD_DATE: str = "Running locally"
        self.IS_PROD = False
        self.IS_DEPLOYED = False
        self.HTTP_PORT: int = 5000
        self.YAML_FILE: str = "api.yaml"

        self.SQL_DB_IP = Props.DEV_SQL_DB_IP
        self.SQL_DB_INSTANCE_NAME = Props.DEV_SQL_DB_INSTANCE_NAME
        self.SQL_DB_USER = Props.SQL_USER
        return self

    def burner(self) -> 'Props':
        self.local()
        self.SQL_DB_IP = Props.BURNER_SQL_DB_IP
        self.SQL_DB_INSTANCE_NAME = Props.BURNER_SQL_DB_INSTANCE_NAME
        self.SQL_DB_USER = Props.SQL_USER
        return self

    def ci(self) -> 'Props':
        self.local()
        self.SQL_DB_IP = Props.CI_SQL_DB_IP
        self.SQL_DB_INSTANCE_NAME = Props.CI_SQL_DB_INSTANCE_NAME
        self.SQL_DB_USER = Props.SQL_USER
        return self

    def prod(self) -> 'Props':
        self.BUILD_DATE = os.environ[
            "BUILD_DATE"] if "BUILD_DATE" in os.environ else "Build date env variable was not set!"
        self.IS_PROD = True
        self.IS_DEPLOYED = True
        self.HTTP_PORT = 80
        self.YAML_FILE = "prod.yaml"

        self.SQL_DB_IP = Props.PROD_SQL_DB_IP
        self.SQL_DB_INSTANCE_NAME = Props.PROD_SQL_DB_INSTANCE_NAME
        self.SQL_DB_USER = Props.SQL_USER
        return self

    def dev(self) -> 'Props':
        self.BUILD_DATE = os.environ[
            "BUILD_DATE"] if "BUILD_DATE" in os.environ else "Build date env variable was not set!"
        self.IS_PROD = False
        self.IS_DEPLOYED = True
        self.HTTP_PORT = 80
        self.YAML_FILE = "dev.yaml"

        self.SQL_DB_IP = Props.DEV_SQL_DB_IP
        self.SQL_DB_INSTANCE_NAME = Props.DEV_SQL_DB_INSTANCE_NAME
        self.SQL_DB_USER = Props.SQL_USER
        return self

    def auto(self) -> 'Props':
        if "CI" in os.environ:
            info("Using CI environment")
            return self.ci()
        is_deployed = "BUILD_DATE" in os.environ
        is_prod = is_deployed and "ELOQUENT_PRODUCTION" in os.environ and os.environ["ELOQUENT_PRODUCTION"] in ["True",
                                                                                                                "true",
                                                                                                                "1", 1]
        if is_deployed:
            if is_prod:
                info("Using PRODUCTION environment")
                self.prod()
            else:
                info("Using DEV environment")
                self.dev()
        else:
            info("Using LOCAL environment")
            if "ELOQUENT_PRODUCTION" in os.environ:
                info("(to mock a production environment, set the BUILD_DATE environment variable)")
            self.local()
        return self


auto: Props = Props().auto()
