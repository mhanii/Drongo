import os
import sys
import base64
from dotenv import load_dotenv

# Add parent directory to path for our module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import real classes
from langchain_google_genai import ChatGoogleGenerativeAI
from ContextStore.context_store import ContextStore

# Import dummy classes for testing
from Agents.ContentAgent.content_ag import ContentAgent
from Agents.ManagerAgent.apply_tool import ApplyTool
from Agents.ManagerAgent.manager_ag import ManagerAgent
# Helper to create a dummy PDF for testing
from fpdf import FPDF

load_dotenv()

class ManagerAgentTester:
    def __init__(self):
        """Initialize the manager agent tester."""
        self.manager_agent = None
        self.store = None
        # Use a real model for the agent's brain, but dummy tools
        self.model = "models/gemini-2.0-flash"

    def print_test_header(self, test_name):
        """Print a formatted test header."""
        print(f"\n{'='*60}")
        print(f"TESTING: {test_name}")
        print(f"{'='*60}")

    def print_test_result(self, result, test_description):
        """Print formatted test result."""
        print(f"\n{test_description}")
        print("-" * 40)
        
        # The result from a ReAct agent is a message object
        if hasattr(result, 'content'):
            print(f"Final Agent Response: {result.content}")
        else:
            print(f"Result: {result}")

    def wait_for_user(self):
        """Wait for user input before continuing."""
        input("\nPress Enter to continue to the next test...")

    def initialize_manager_agent(self):
        """Initialize the Manager Agent with dummy tools for testing."""
        print("Initializing Manager Agent Test Suite...")
        
        self.store = ContextStore(max_window=10)
        
        # IMPORTANT: We instantiate our dummy tools here
        dummy_content_agent = ContentAgent(
            model="models/gemini-2.0-flash",
            # store=self.store,
            checkpoint_path="data/checkpoint.sqlite"
        )
        dummy_apply_tool = ApplyTool(content_db=dummy_content_agent.chunk_db)

        # Instantiate the real ManagerAgent
        self.manager_agent = ManagerAgent(
            model=self.model,
            store=self.store,
            checkpoint_path="data/manager_checkpoint.sqlite"
        )
        
        # Override the real tools with our dummy ones
        self.manager_agent.content_agent = dummy_content_agent
        self.manager_agent.apply_tool = dummy_apply_tool
        
        print("Manager Agent initialized successfully with DUMMY tools!")
        return True

    def test_simple_text_generation(self):
        """Test a basic text generation request without any external context."""
        self.print_test_header("SIMPLE TEXT GENERATION")

        request_data = {
            "text": "Add a summary at the end of the document",
            "document_structure": self.get_document_structure(),
            "images": [],
            "documents": []
        }
        
        print("Running a simple text generation task...")
        print("Expected behavior: Agent should call 'generate_content' then 'apply_tool_func'.")
        
        result = self.manager_agent.run_prompt(request_data)
        self.print_test_result(result, "Result of simple text generation task")
        self.wait_for_user()


    def get_document_structure(self):
        return """BEGINNING OF DOCUMENT:
<p position-id="0"><span style="font-weight: bold;font-size: 20PT;font-family: "Pacifico";color: rgb(60, 120, 216);">SWIFTSTAY<br></span></p>
<p position-id="1"><span style="font-weight: bold;font-size: 18PT;font-family: "Microsoft YaHei UI";">Practica1<br></span></p>
<p position-id="2"><span style="font-weight: bold;font-size: 15PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Descripción del sistema en lenguaje natural<br></span></p>
<p position-id="3"><span style="font-family: "Montserrat";">El sistema de gestión de hotel es una plataforma que facilita la administración y operación eficiente del hotel. Incluye varios subsistemas que colaboran para proporcionar funcionalidad completa para huéspedes y personal. El subsistema de Gestión de Habitaciones mantiene un registro detallado de cada habitación, almacenando información clave como número, tipo, tarifa y disponibilidad, además de gestionar servicios adicionales, las asignaciones y generar alertas de disponibilidad. El Proceso de Reserva permite a los usuarios buscar, seleccionar y reservar habitaciones, mostrando información detallada y facilitando check-ins, modificaciones y check-outs. La Administración de Cuentas de Usuario gestiona la creación, autenticación y cierre de cuentas, almacenando y modificando información personal, y permitiendo ver historiales sobre las reservas hechas de un usuario.. El subsistema de Facturación se encarga de procesar pagos, generar facturas detalladas y mantener un historial de transacciones. <br></span></p>
<p position-id="6"><span style="font-weight: bold;font-family: "Montserrat";color: rgb(109, 158, 235);">Gestión de habitaciones (Omar)</span><span style="font-family: "Montserrat";color: rgb(109, 158, 235);">:<br></span></p>
<p position-id="7"><span style="font-family: "Montserrat";">Este subsistema contiene el registro de cada habitación del hotel e información sobre estas: el número de la habitación, el tipo de habitación, la tarifa por noche por tipo de habitación, la disponibilidad. las comodidades e información del huésped. Se encarga de la gestión de otros servicios de coste adicional (nombre del servicio, descripción, precio del servicio). Genera alertas y notificaciones en caso de que la solicitud de las habitaciones de un tipo estén por debajo de un umbral definido (umbral, tipo de habitación). Se puede mostrar también la información de cada habitación (se muestra cada registro) y en caso de estar ocupada por un usuario mostrar información sobre este (nº de habitación)<br></span></p>
<p position-id="10"><span style="font-weight: bold;font-family: "Montserrat";color: rgb(109, 158, 235);">Proceso de reserva (Manuel)</span><span style="font-family: "Montserrat";color: rgb(109, 158, 235);">:<br></span></p>
<p position-id="11"><span style="font-family: "Montserrat";">Este permite a los usuarios buscar, elegir y reservar habitaciones de hotel según sus preferencias, incluido el tipo de habitación, las fechas de entrada y salida, la cantidad de adultos y la cantidad de niños. Se muestran todos los detalles de las habitaciones disponibles, incluido el precio por noche, la disponibilidad, descripciones detalladas y fotografías. Al proporcionar información como su identificación, el número de habitación deseado y las fechas de su estadía, los usuarios pueden realizar reservas. A continuación recibirán un correo electrónico de confirmación con todos los detalles de su reserva. El sistema permite el check-in proporcionando identificación, número de reserva y fecha de llegada, y permite modificaciones de la reserva dentro de ciertos límites y restricciones. También genera una confirmación con la fecha y hora del check-in. Además, simplifica el check-out utilizando el número de confirmación de la reserva, la fecha de salida y el método de pago enviando una confirmación que incluye la fecha y hora del check-out, el monto total adeudado y el estado de la habitación. Para garantizar que las fechas sean precisas, las salas estén disponibles y los datos del usuario sean distintos y válidos en el sistema, se han implementado restricciones semánticas.<br></span></p>
<p position-id="13"><span style="font-weight: bold;font-family: "Montserrat";color: rgb(109, 158, 235);">Administración de cuentas de usuario (ZhanyongPan)</span><span style="font-family: "Montserrat";color: rgb(109, 158, 235);">:<br></span></p>
<p position-id="14"><span style="font-family: "Montserrat";">Este se trata del subsistema para crear cuentas y permitir a los usuarios registrados acceder, gestionar y personalizar su experiencia dentro del sistema.<br></span></p>
<p position-id="15"><span style="font-family: "Montserrat";">Este subsistema se encarga de la creación de cuentas de usuario, la autenticación segura, la gestión de perfiles de usuario, el seguimiento de historiales de reservas y la baja de cuenta. Para crear una cuenta de usuario se almacena el nombre de cuenta y la contraseña que se proporcione, estos se utilizan para iniciar sesión como usuario en nuestra plataforma. De cada usuario se almacena su nombre, el DNI, el número de teléfono, el correo electrónico, la localidad y la dirección postal. Se confirman las inserciones o se avisa en caso de error. Un usuario solo puede tener un correo y un número de teléfono, y ambos solo pueden pertenecer a un solo usuario. El subsistema permitirá mostrar un listado con todas las reservas realizadas por cada usuario almacenando el código de reserva, las fechas de estancia, el tipo de habitación y el precio de la reserva . El usuario podrá dar de baja su cuenta siempre y cuando quiera introduciendo su cuenta y contraseña.<br></span></p>
<p position-id="18"><span style="font-weight: bold;font-family: "Montserrat";color: rgb(109, 158, 235);">Facturación (Mohamed Hani)</span><span style="font-family: "Montserrat";color: rgb(109, 158, 235);">:<br></span></p>
<p position-id="19"><span style="font-family: "Montserrat";">Los clientes pueden pagar sus reservas de varias formas en este sistema, como por ejemplo con tarjetas de crédito o transferencias bancarias. Por lo tanto, deberán proporcionar los datos de la tarjeta de crédito como número de tarjeta de crédito, fecha de vencimiento y cvv. El coste de la habitación por noche y el importe total de la estancia se necesitan de otras funcionalidades para calcular el coste total.<br></span></p>
<p position-id="20"><span style="font-family: "Montserrat";">Los clientes pueden hacer "facturación dividida" con sus amigos o familiares. Los clientes también pueden obtener un reembolso en caso de cancelar la reserva si tienen una reserva premium o en caso de situaciones imprevistas. <br></span></p>
<p position-id="21"><span style="font-family: "Montserrat";">El sistema genera facturas por las reservas realizadas que incluyen el precio del hotel, impuestos y cualquier otro cargo. Por lo tanto, se debe proporcionar una dirección de correo electrónico y un número de teléfono. También son necesarios el nombre, apellido y número de habitación, que se pueden recuperar desde otras funcionalidades. Las facturas por correo electrónico a los clientes se entregan automáticamente una vez que se completa el procedimiento de pago. Finalmente, tanto los clientes como los administradores tienen acceso a un historial de transacciones que muestra todas las facturas emitidas y los pagos registrados.</span><br></p>
<p position-id="23"><span style="font-weight: bold;font-style: italic;font-size: 20PT;font-family: "Montserrat";color: rgb(17, 85, 204);">Gestión de habitaciones </span><span style="font-weight: bold;font-style: italic;text-decoration: underline;font-size: 20PT;font-family: "Montserrat";color: rgb(17, 85, 204);">(Omar)</span><span style="font-weight: bold;font-style: italic;font-size: 20PT;font-family: "Montserrat";color: rgb(17, 85, 204);">:<br></span></p>
<p position-id="25"><span style="font-weight: bold;text-decoration: underline;font-size: 18PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Requisitos funcionales</span><span style="font-size: 18PT;font-family: "Montserrat";">:<br></span></p>
<p position-id="27"><span style="font-weight: bold;text-decoration: underline;font-size: 17PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RF-1.1 </span><span style="font-weight: bold;text-decoration: underline;font-size: 16PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Registrar habitación:</span><span style="font-weight: bold;text-decoration: underline;font-size: 14PT;font-family: "Montserrat";color: rgb(109, 158, 235);"><br></span></p>
<p position-id="28"><span style="font-size: 12PT;font-family: "Montserrat";">El sistema mantendrá un registro con todas las habitaciones. Este registro contiene información básica de las habitaciones  como su número, el tipo de habitación y la tarifa por noche de cada habitación, la disponibilidad de la habitación y ciertos datos del huésped.<br></span></p>
<p position-id="30"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Entrada</span><span style="font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">:<br></span></p>
<p position-id="31"><span style="font-family: "Montserrat";">	</span><span style="font-weight: bold;font-style: italic;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Agente externo:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Administrador<br></span></p>
<p position-id="32"><span style="font-size: 12PT;font-family: "Montserrat";">	</span><span style="font-weight: bold;font-style: italic;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Acción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Solicitar añadir información sobre una habitación	</span><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de entrada:</span><span style="font-size: 12PT;font-family: "Montserrat";"> (RDE-1.1)<br></span></p>
<p position-id="34"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">BD:<br></span></p>
<p position-id="35"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de entrada:</span><span style="font-size: 12PT;font-family: "Montserrat";"> (RDW-1.1)<br></span></p>
<p position-id="37"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Salida:</span><span style="font-size: 12PT;font-family: "Montserrat";"><br></span></p>
<p position-id="38"><span style="font-size: 12PT;font-family: "Montserrat";">	</span><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Agente externo:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Administrador<br></span></p>
<p position-id="39"><span style="font-size: 12PT;font-family: "Montserrat";">	</span><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Acción</span><span style="font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Confirmación de habitación añadida.<br></span></p>
<p position-id="42"><span style="font-weight: bold;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDE-1.1: </span><span style="font-size: 13PT;font-family: "Montserrat";">Datos de la habitación</span><span style="font-family: "Montserrat";"><br></span></p>
<p position-id="43"><span style="font-family: "Montserrat";">-</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">Número de habitación</span><span style="font-family: "Montserrat";">: int (3).<br></span></p>
<p position-id="44"><span style="font-family: "Montserrat";">-</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">Tipo de habitación:</span><span style="font-family: "Montserrat";"> Cadena de caracteres (30).<br></span></p>
<p position-id="45"><span style="font-family: "Montserrat";">-</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">Precio/noche:</span><span style="font-family: "Montserrat";"> int (4).<br></span></p>
<p position-id="46"><span style="font-family: "Montserrat";">-</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">Disponibilidad</span><span style="font-family: "Montserrat";">: bool.<br></span></p>
<p position-id="49"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDW-1.1</span><span style="font-weight: bold;text-decoration: underline;font-family: "Montserrat";color: rgb(109, 158, 235);">: </span><span style="font-size: 13PT;font-family: "Montserrat";">Datos de la habitación</span><span style="font-weight: bold;text-decoration: underline;font-family: "Montserrat";color: rgb(109, 158, 235);"><br></span></p>
<p position-id="50"><span style="font-family: "Montserrat";">-</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">Número de habitación</span><span style="font-family: "Montserrat";">: Cadena de int (3).<br></span></p>
<p position-id="51"><span style="font-family: "Montserrat";">-</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">Tipo de habitación:</span><span style="font-family: "Montserrat";"> Cadena de caracteres(30).<br></span></p>
<p position-id="52"><span style="font-family: "Montserrat";">-</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">Precio/noche:</span><span style="font-family: "Montserrat";"> Cadena de int (4).<br></span></p>
<p position-id="53"><span style="font-family: "Montserrat";">-</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">Disponibilidad</span><span style="font-family: "Montserrat";">: bool.<br></span></p>
<p position-id="56"><span style="font-weight: bold;text-decoration: underline;font-size: 15PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Restricciones semánticas(RF-1.1)<br></span></p>
<p position-id="57"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(0, 255, 0);">RS-1:</span><span style="font-size: 12PT;font-family: "Montserrat";"> El número de la habitación debe ser único.<br></span></p>
<p position-id="58"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(255, 0, 0);">RF:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Registrar habitación (RF-1.1)<br></span></p>
<p position-id="59"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">RD(s)</span><span style="font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">:</span><span style="font-size: 12PT;font-family: "Montserrat";"> RDE-1.1<br></span></p>
<p position-id="60"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";">Descripción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> "Si se intenta registrar una habitación con un número que ya existe en el sistema, se devolverá un error."<br></span></p>
<p position-id="63"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(0, 255, 0);">RS-2:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Comprobar disponibilidad de habitación.<br></span></p>
<p position-id="64"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(255, 0, 0);">RF:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Registrar habitación (RF-1.1)<br></span></p>
<p position-id="65"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">RD(s)</span><span style="font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">:</span><span style="font-size: 12PT;font-family: "Montserrat";"> RDE-1.1<br></span></p>
<p position-id="66"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";">Descripción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> "Antes de registrar una nueva habitación, el sistema debe verificar si el número de habitación ya existe en el sistema para evitar duplicados."<br></span></p>
<p position-id="69"><span style="font-weight: bold;text-decoration: underline;font-size: 16PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RF-1.2. Gestionar servicios adicionales</span><span style="font-family: "Montserrat";"><br></span></p>
<p position-id="70"><span style="font-size: 12PT;font-family: "Montserrat";">El sistema permite a los administradores definir y actualizar los servicios adicionales disponibles para los huéspedes, así como sus respectivos precios. Esto asegura que los servicios estén correctamente configurados y disponibles para su selección durante el proceso de reserva.<br></span></p>
<p position-id="72"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Entrada</span><span style="font-size: 14PT;font-family: "Montserrat";color: rgb(109, 158, 235);">:<br></span></p>
<p position-id="73"><span style="font-weight: bold;font-style: italic;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Agente externo:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Administrador<br></span></p>
<p position-id="74"><span style="font-weight: bold;font-style: italic;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Acción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Actualizar servicios adicionales	</span><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de entrada:</span><span style="font-size: 12PT;font-family: "Montserrat";"> </span><span style="font-family: "Montserrat";">(RDE-1.2)</span><span style="font-size: 10PT;font-family: "Montserrat";"><br></span></p>
<p position-id="77"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">DB:<br></span></p>
<p position-id="78"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de escritura:</span><span style="font-size: 12PT;font-family: "Montserrat";"> (RDW-1.2)</span><span style="font-family: "Montserrat";"><br></span></p>
<p position-id="80"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Salida:</span><span style="font-family: "Montserrat";"> </span><span style="font-size: 14PT;font-family: "Montserrat";color: rgb(109, 158, 235);"><br></span></p>
<p position-id="81"><span style="font-weight: bold;font-style: italic;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Agente externo:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Administrador<br></span></p>
<p position-id="82"><span style="font-weight: bold;font-style: italic;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Acción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Servicios adicionales actualizados<br></span></p>
<p position-id="84"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDE-1.2: </span><span style="font-size: 13PT;font-family: "Montserrat";">Datos para gestionar servicios adicionales</span><span style="font-family: "Montserrat";"><br></span></p>
<p position-id="85"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Nombre del servicio: </span><span style="font-family: "Montserrat";">Cadena de caracteres (50)<br></span></p>
<p position-id="86"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Descripción del servicio: </span><span style="font-family: "Montserrat";">Cadena de caracteres (100)<br></span></p>
<p position-id="87"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Precio del servicio: </span><span style="font-family: "Montserrat";"> int(4).</span><span style="font-weight: bold;font-family: "Montserrat";color: rgb(164, 194, 244);"><br></span></p>
<p position-id="89"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDW-1.2: <br></span></p>
<p position-id="90"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Nombre del servicio: </span><span style="font-family: "Montserrat";">Cadena de caracteres (50)<br></span></p>
<p position-id="91"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Descripción del servicio: </span><span style="font-family: "Montserrat";">Cadena de caracteres (100)<br></span></p>
<p position-id="92"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Precio del servicio: </span><span style="font-family: "Montserrat";"> int(4).</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);"><br></span></p>
<p position-id="94"><span style="font-weight: bold;text-decoration: underline;font-size: 16PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Requisitos semánticos(RF-1.2)<br></span></p>
<p position-id="95"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(0, 255, 0);">RS-3:</span><span style="font-size: 12PT;font-family: "Montserrat";color: rgb(0, 255, 0);"> </span><span style="font-size: 12PT;font-family: "Montserrat";">Verificación de servicio único<br></span></p>
<p position-id="96"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(255, 0, 0);">RF:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Gestionar servicios adicionales (RF-1.2)<br></span></p>
<p position-id="97"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">RD(s)</span><span style="font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">:</span><span style="font-size: 12PT;font-family: "Montserrat";"> RDE-1.2<br></span></p>
<p position-id="98"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";">Descripción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> "El sistema debe verificar que el nombre del servicio proporcionado sea único y no esté duplicado en el sistema. Si se intenta agregar un servicio con un nombre ya existente, se debe devolver un error."<br></span></p>
<p position-id="100"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(0, 255, 0);">RS-4:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Actualización de servicios exitosa.<br></span></p>
<p position-id="101"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(255, 0, 0);">RF:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Gestionar servicios adicionales (RF-1.2)<br></span></p>
<p position-id="102"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">RD(s)</span><span style="font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">:</span><span style="font-size: 12PT;font-family: "Montserrat";"> RDE-1.2<br></span></p>
<p position-id="103"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";">Descripción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> "El sistema debe asegurarse de que los servicios adicionales se actualicen correctamente en la base de datos después de definirlos o actualizarlos. Si ocurre un error durante la actualización, se debe notificar al usuario."<br></span></p>
<p position-id="106"><span style="font-weight: bold;text-decoration: underline;font-size: 16PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RF-1.3. Modificar tipo de habitación:</span><span style="font-family: "Montserrat";"><br></span></p>
<p position-id="107"><span style="font-size: 12PT;font-family: "Montserrat";">El sistema permite a los administradores modificar los tipos de habitación y los datos asociados a ellos.<br></span></p>
<p position-id="109"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Entrada:<br></span></p>
<p position-id="110"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Agente externo:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Administrador<br></span></p>
<p position-id="111"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Acción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Solicitar modificación de tipo de habitación.<br></span></p>
<p position-id="112"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de entrada:</span><span style="font-size: 12PT;font-family: "Montserrat";"> (RDE-1.3)<br></span></p>
<p position-id="114"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">DB:<br></span></p>
<p position-id="115"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de escritura:</span><span style="font-size: 12PT;font-family: "Montserrat";"> (RDW-1.3)<br></span></p>
<p position-id="117"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Salida:<br></span></p>
<p position-id="118"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Agente externo:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Administrador<br></span></p>
<p position-id="119"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Acción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Tipo de habitación modificado.<br></span></p>
<p position-id="121"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDE-1.3</span><span style="font-size: 12PT;font-family: "Montserrat";">: Datos de tipo de habitación<br></span></p>
<p position-id="122"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Tipo de habitación:</span><span style="font-family: "Montserrat";"> Cadena de caracteres (30).<br></span></p>
<p position-id="123"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Precio/noche:</span><span style="font-family: "Montserrat";"> Cadena de int (4).<br></span></p>
<p position-id="125"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDW-1.3:<br></span></p>
<p position-id="126"><span style="font-size: 12PT;font-family: "Montserrat";">	</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Tipo de habitación:</span><span style="font-family: "Montserrat";"> Cadena de caracteres (30).<br></span></p>
<p position-id="127"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Precio/noche:</span><span style="font-family: "Montserrat";"> Cadena de int (4).</span><span style="font-size: 12PT;font-family: "Montserrat";"><br></span></p>
<p position-id="129"><span style="font-weight: bold;text-decoration: underline;font-size: 16PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Requisitos semánticos(RF-1.3)</span><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);"><br></span></p>
<p position-id="131"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(0, 255, 0);">RS-5:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Verificación de asignación correcta.<br></span></p>
<p position-id="132"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(255, 0, 0);">RF:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Modificar tipo de habitación (RF-1.3)<br></span></p>
<p position-id="133"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">RD(s)</span><span style="font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">:</span><span style="font-size: 12PT;font-family: "Montserrat";"> RDE-1.3<br></span></p>
<p position-id="134"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";">Descripción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> "El sistema debe verificar si las modificaciones han sido correctamente registradas."<br></span></p>
<p position-id="136"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(0, 255, 0);">RS-6:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Verificación de tipo de habitación.<br></span></p>
<p position-id="137"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(255, 0, 0);">RF:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Modificar tipo de habitación (RF-1.3)<br></span></p>
<p position-id="138"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">RD(s):</span><span style="font-size: 12PT;font-family: "Montserrat";"> RDE-1.3<br></span></p>
<p position-id="139"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";">Descripción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> “El sistema debe comprobar que el tipo que se va a modificar existe previamente. En caso contrario, se genera un error.”<br></span></p>
<p position-id="160"><span style="font-weight: bold;text-decoration: underline;font-size: 16PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RF-1.4. Generar alertas de baja demanda</span><span style="font-family: "Montserrat";"><br></span></p>
<p position-id="161"><span style="font-size: 12PT;font-family: "Montserrat";">El sistema genera alertas y notificaciones cuando la demanda de las habitaciones de un tipo está por debajo de un umbral definido. La generación de alertas se lleva a cabo de forma automática en intervalos regulares de un mes en un proceso programado en el sistema.Los administradores del sistema tienen la capacidad de definir y ajustar los parámetros de generación de alertas y estos son introducidos en su correspondiente interfaz para establecer el umbral de cada tipo de habitación.<br></span></p>
<p position-id="163"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Entrada:<br></span></p>
<p position-id="164"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Agente externo:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Sistema<br></span></p>
<p position-id="165"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Acción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Verificar demanda y generar alerta si es necesario<br></span></p>
<p position-id="166"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de entrada:</span><span style="font-size: 12PT;font-family: "Montserrat";"> (RDE-1.4)<br></span></p>
<p position-id="168"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">DB</span><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(109, 158, 235);">:<br></span></p>
<p position-id="169"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de lectura:</span><span style="font-size: 12PT;font-family: "Montserrat";">(RDR-1.4)<br></span></p>
<p position-id="171"><span style="font-weight: bold;text-decoration: underline;font-size: 12PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Salida:</span><span style="font-weight: bold;text-decoration: underline;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);"><br></span></p>
<p position-id="172"><span style="font-weight: bold;font-style: italic;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Agente Externo</span><span style="font-style: italic;font-size: 12PT;font-family: "Montserrat";">:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Sistema<br></span></p>
<p position-id="173"><span style="font-weight: bold;font-style: italic;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Acción: </span><span style="font-size: 12PT;font-family: "Montserrat";">Alerta generada<br></span></p>
<p position-id="174"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de salida: </span><span style="font-size: 12PT;font-family: "Montserrat";">(RDS-1.4)<br></span></p>
<p position-id="176"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDE-1.4.</span><span style="font-size: 12PT;font-family: "Montserrat";">: Datos para generar alertas: <br></span></p>
<p position-id="177"><span style="font-size: 12PT;font-family: "Montserrat";">	</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Umbral de disponibilidad:</span><span style="font-family: "Montserrat";"> int (1-100).<br></span></p>
<p position-id="178"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Tipo de habitación: </span><span style="font-family: "Montserrat";">Cadena de caracteres (30)<br></span></p>
<p position-id="181"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDR-1.4.</span><span style="font-size: 12PT;font-family: "Montserrat";">:<br></span></p>
<p position-id="182"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Umbral de disponibilidad:</span><span style="font-family: "Montserrat";"> int (1-100).<br></span></p>
<p position-id="183"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Tipo de habitación: </span><span style="font-family: "Montserrat";">Cadena de caracteres (30)</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);"><br></span></p>
<p position-id="187"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDS-1.4.</span><span style="font-size: 12PT;font-family: "Montserrat";">:<br></span></p>
<p position-id="188"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Alerta generada: </span><span style="font-family: "Montserrat";">Cadena de caracteres (100).</span><span style="font-family: "Montserrat";color: rgb(164, 194, 244);"><br></span></p>
<p position-id="191"><span style="font-weight: bold;text-decoration: underline;font-size: 16PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Requisitos semánticos(RF-1.4)</span><span style="font-size: 12PT;font-family: "Montserrat";"><br></span></p>
<p position-id="193"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(0, 255, 0);">RS-8:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Umbral de disponibilidad válido<br></span></p>
<p position-id="194"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(255, 0, 0);">RF:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Generar alertas de baja demanda (RF-1.4)<br></span></p>
<p position-id="195"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">RD(s)</span><span style="font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">:</span><span style="font-size: 12PT;font-family: "Montserrat";"> RDE-1.4.<br></span></p>
<p position-id="196"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";">Descripción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> "El sistema debe verificar si el umbral de demanda proporcionado es un valor válido (entre 1 y 100). Si no lo es, se debe devolver un error."<br></span></p>
<p position-id="200"><span style="font-weight: bold;text-decoration: underline;font-size: 16PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RF-1.5. Gestionar estado de habitaciones </span><span style="font-family: "Montserrat";"><br></span></p>
<p position-id="201"><span style="font-size: 12PT;font-family: "Montserrat";">El sistema permite a los administradores cambiar el estado de una habitación a "En Mantenimiento" cuando sea necesario realizar reparaciones o mejoras. Esto asegura que la habitación no esté disponible para reserva hasta que el mantenimiento se complete y el estado se actualice.<br></span></p>
<p position-id="203"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Entrada:<br></span></p>
<p position-id="204"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Agente externo:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Administrador<br></span></p>
<p position-id="205"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Acción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Cambiar estado de una habitación en mantenimiento<br></span></p>
<p position-id="206"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de entrada:</span><span style="font-size: 12PT;font-family: "Montserrat";"> (RDE-1.5.)<br></span></p>
<p position-id="208"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">DB</span><span style="font-weight: bold;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">:<br></span></p>
<p position-id="209"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Requisito de datos de lectura</span><span style="font-size: 12PT;font-family: "Montserrat";">: (RDW-1.5.)<br></span></p>
<p position-id="211"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Salida:</span><span style="font-weight: bold;text-decoration: underline;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);"><br></span></p>
<p position-id="212"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Agente Externo:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Administrador<br></span></p>
<p position-id="213"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(164, 194, 244);">Acción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Estado de habitación actualizado<br></span></p>
<p position-id="215"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDE-1.5.:</span><span style="font-size: 13PT;font-family: "Montserrat";"> Datos para gestionar estado de habitaciones para mantenimiento:<br></span></p>
<p position-id="216"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Número de habitación:</span><span style="font-family: "Montserrat";"> Cadena de int (3)</span><span style="font-size: 13PT;font-family: "Montserrat";">.<br></span></p>
<p position-id="217"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Nuevo estado: </span><span style="font-family: "Montserrat";">Lista (Disponible, En Mantenimiento, Limpieza)</span><span style="font-size: 13PT;font-family: "Montserrat";"><br></span></p>
<p position-id="218"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Motivo de mantenimiento:</span><span style="font-family: "Montserrat";"> Campo de texto (100)</span><span style="font-size: 13PT;font-family: "Montserrat";">.<br></span></p>
<p position-id="220"><span style="font-weight: bold;text-decoration: underline;font-size: 13PT;font-family: "Montserrat";color: rgb(109, 158, 235);">RDW-1.5.: <br></span></p>
<p position-id="221"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Número de habitación:</span><span style="font-family: "Montserrat";"> Cadena de int (3)</span><span style="font-size: 13PT;font-family: "Montserrat";">.<br></span></p>
<p position-id="222"><span style="font-family: "Montserrat";color: rgb(164, 194, 244);">-Nuevo estado: </span><span style="font-family: "Montserrat";">Campo de texto (30)</span><span style="font-size: 13PT;font-family: "Montserrat";">.</span><span style="font-size: 12PT;font-family: "Montserrat";"><br></span></p>
<p position-id="227"><span style="font-weight: bold;text-decoration: underline;font-size: 16PT;font-family: "Montserrat";color: rgb(109, 158, 235);">Requisitos semánticos(RF-1.5)</span><span style="font-size: 12PT;font-family: "Montserrat";"><br></span></p>
<p position-id="229"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(0, 255, 0);">RS-9:</span><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(109, 158, 235);"> </span><span style="font-size: 12PT;font-family: "Montserrat";">Motivo de mantenimiento obligatorio<br></span></p>
<p position-id="230"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(255, 0, 0);">RF:</span><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(109, 158, 235);"> </span><span style="font-size: 12PT;font-family: "Montserrat";">Gestionar estado de habitaciones (RF-1.5)<br></span></p>
<p position-id="231"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">RD(s)</span><span style="font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">:</span><span style="font-size: 12PT;font-family: "Montserrat";"> RDE-1.5<br></span></p>
<p position-id="232"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";">Descripción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> "El administrador debe proporcionar un motivo de mantenimiento al cambiar el estado de una habitación a “En Mantenimiento". Si no se proporciona, se debe devolver un error."<br></span></p>
<p position-id="234"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(0, 255, 0);">RS-10:</span><span style="font-size: 12PT;font-family: "Montserrat";"> Estado de habitación actualizado correctamente<br></span></p>
<p position-id="235"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(255, 0, 0);">RF:</span><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(109, 158, 235);"> </span><span style="font-size: 12PT;font-family: "Montserrat";">Gestionar estado de habitaciones (RF-1.5)<br></span></p>
<p position-id="236"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">RD(s)</span><span style="font-size: 12PT;font-family: "Montserrat";color: rgb(17, 85, 204);">:</span><span style="font-size: 12PT;font-family: "Montserrat";"> RDE-1.5.<br></span></p>
<p position-id="237"><span style="font-weight: bold;font-size: 12PT;font-family: "Montserrat";">Descripción:</span><span style="font-size: 12PT;font-family: "Montserrat";"> "Después de cambiar el estado de una habitación a "En Mantenimiento", el sistema debe asegurarse de que el nuevo estado se refleje correctamente en el registro de habitaciones."<br></span></p>
END OF DOCUMENT"""


    def test_text_generation_with_pdf_context(self):
        """Test text generation that requires context from an uploaded PDF."""
        self.print_test_header("TEXT GENERATION WITH PDF CONTEXT")

        # 1. Create a dummy PDF on the fly
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Project Gemini: Annual Report", ln=True, align='C')
        pdf.multi_cell(0, 10, txt="In 2024, Project Gemini achieved a 200% increase in user engagement. Key factors included the new 'ManagerAgent' and 'ContentAgent' modules, which streamlined document editing.")
        pdf.output("dummy_report.pdf")

        # 2. Read and encode the PDF
        with open("dummy_report.pdf", "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        os.remove("dummy_report.pdf") # Clean up

        request_data = {
            "text": "Please read the attached report and write a two-sentence summary to be placed at the very top of the document.",
            "document_structure": "<html><body><h1>Internal Memo</h1></body></html>",
            "images": [],
            "documents": [{
                "name": "dummy_report.pdf",
                "content": pdf_b64
            }]
        }
        
        print("Running a task that requires reading a PDF...")
        print("Expected behavior: Agent should process the PDF and use its content in the prompt to 'generate_content'.")

        result = self.manager_agent.run_prompt(request_data)
        self.print_test_result(result, "Result of text generation with PDF context")
        self.wait_for_user()
    
    def test_error_handling(self):
        """Test how the agent handles a request outside its scope."""
        self.print_test_header("ERROR HANDLING (Out of Scope Request)")
        
        request_data = {
            "text": "What is the current weather in London? Don't use your tools, just answer me.",
            "document_structure": "<html><body><p>Some text.</p></body></html>"
        }

        print("Running a task that is outside the agent's defined scope...")
        print("Expected behavior: Agent should state that it cannot fulfill the request as it's a document editor.")

        result = self.manager_agent.run_prompt(request_data)
        self.print_test_result(result, "Result of out-of-scope request")
        self.wait_for_user()

    def run_interactive_mode(self):
        """Run interactive mode for manual testing."""
        self.print_test_header("INTERACTIVE MODE")
        print("Interactive testing mode - try any document editing command!")
        print("   Type 'quit' or 'exit' to end.")
        
        doc_structure = "<html><body><h1>Test Document</h1><p>This is the first paragraph.</p><h2>Conclusion</h2><p>The end.</p></body></html>"
        print("\nInitial Document Structure:")
        print(doc_structure)
        
        while True:
            try:
                prompt = input("\nEnter command: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue
                
                print(f"\n🔄 Processing: {prompt}")
                request_data = {
                    "text": prompt,
                    "document_structure": doc_structure
                }
                result = self.manager_agent.run_prompt(request_data)
                self.print_test_result(result, "Interactive Command Result")
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def run_all_tests(self):
        """Run the full test suite."""
        try:
            if not self.initialize_manager_agent():
                return

            self.test_simple_text_generation()
            # self.test_text_generation_with_pdf_context()
            self.test_error_handling()
            self.run_interactive_mode()

            print("\n🎉 Manager Agent test suite completed!")

        except Exception as e:
            print(f"A critical error occurred during the test suite: {e}")

if __name__ == "__main__":
    tester = ManagerAgentTester()
    tester.run_all_tests()