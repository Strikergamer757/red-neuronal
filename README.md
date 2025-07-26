def resumir_texto(self, texto: str, num_sentencias: int = 5, idioma: str = "spanish") -> str:
        try:
            if not texto.strip():
                return "No hay contenido para resumir."
            
            texto_limpio = self._limpiar_texto(texto)
            
            if len(texto_limpio) > self.configuracion['fragmento_max']:
                texto_limpio = texto_limpio[:self.configuracion['fragmento_max']]
            
            if len(texto_limpio) < 200:
                return texto_limpio
            
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lsa import LsaSummarizer
            
            parser = PlaintextParser.from_string(texto_limpio, Tokenizer(idioma))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, num_sentencias)
            
            resumen = " ".join(str(sentence) for sentence in summary)
            resumen_mejorado = self._reformular_resumen(resumen)
            
            return resumen_mejorado if resumen_mejorado.strip() else texto_limpio[:500] + "..."
            
        except Exception as e:
            logger.error(f"Error al resumir: {e}")
            return f"Error al resumir el texto: {str(e)}"

    def _reformular_resumen(self, resumen: str) -> str:
        """Mejora la calidad del resumen con reformulaciones"""
        reformulaciones = {
            r'según\s+(.+?),': r'De acuerdo con \1, se establece que',
            r'es\s+importante\s+destacar\s+que': r'Cabe resaltar que',
            r'por\s+lo\s+tanto': r'En consecuencia',
            r'además': r'Asimismo',
            r'sin\s+embargo': r'No obstante',
            r'en\s+conclusión': r'Para concluir',
            r'esto\s+significa\s+que': r'Lo anterior implica que'
        }
        
        resumen_reformulado = resumen
        for patron, reemplazo in reformulaciones.items():
            resumen_reformulado = re.sub(patron, reemplazo, resumen_reformulado, flags=re.IGNORECASE)
        
        oraciones = resumen_reformulado.split('. ')
        if len(oraciones) > 1:
            conectores = ['Por otra parte', 'De igual manera', 'En este sentido', 'Adicionalmente']
            for i in range(1, len(oraciones)):
                if i % 2 == 0 and len(oraciones[i]) > 20:
                    conector = conectores[i % len(conectores)]
                    oraciones[i] = f"{conector}, {oraciones[i].lower()}"
        
        return '. '.join(oraciones).strip()

    def _limpiar_texto(self, texto: str) -> str:
        """Limpia y normaliza el texto"""
        texto = re.sub(r'\s+', ' ', texto)
        texto = re.sub(r'[^\w\s.,;:!?¿¡\-áéíóúñüÁÉÍÓÚÑÜ()\"\'\/]', '', texto)
        return texto.strip()

    def resumir_largo(self, texto: str, num_sentencias: int = 5) -> str:
        """Resumir textos largos iterativamente"""
        fragmento_max = self.configuracion['fragmento_max']
        max_iter = self.configuracion['max_iteraciones']
        
        resumen_total = texto
        
        for iteracion in range(max_iter):
            if len(resumen_total) <= fragmento_max:
                break
            
            parrafos = resumen_total.split('\n\n')
            if len(parrafos) > 1 and all(len(p) < fragmento_max for p in parrafos):
                fragmentos = parrafos
            else:
                fragmentos = [resumen_total[i:i+fragmento_max] 
                             for i in range(0, len(resumen_total), fragmento_max)]
            
            resumenes = []
            for frag in fragmentos:
                if frag.strip():
                    resumen_frag = self.resumir_texto(frag, num_sentencias)
                    resumenes.append(resumen_frag)
            
            resumen_total = " ".join(resumenes)
            
            if iteracion > 0 and len(resumen_total) >= len(texto) * 0.8:
                break
        
        return resumen_total

    def buscar_imagenes_google(self, query: str, num: int = 2) -> List[str]:
        if not self.serpapi_key or self.serpapi_key == "YOUR_SERPAPI_KEY":
            logger.warning("API Key de SerpAPI no configurada")
            return []
        
        cache_key = f"img_{query}_{num}"
        resultado_cache = self.cache_busquedas.get(cache_key)
        if resultado_cache:
            return resultado_cache
        
        try:
            from serpapi import GoogleSearch
            params = {
                "engine": "google",
                "q": query,
                "tbm": "isch",
                "api_key": self.serpapi_key,
                "ijn": 0,
                "num": min(num, self.configuracion['max_imagenes']),
                "safe": "active"
            }
            search_instance = GoogleSearch(params)
            results = search_instance.get_dict()
            
            imagenes = []
            if "images_results" in results:
                for img in results["images_results"][:num]:
                    if "original" in img and self._es_imagen_valida(img["original"]):
                        imagenes.append(img["original"])
            
            self.cache_busquedas.set(cache_key, imagenes)
            return imagenes
            
        except Exception as e:
            logger.error(f"Error al buscar imágenes: {e}")
            return []

    def _es_imagen_valida(self, url: str) -> bool:
        extensiones_validas = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']
        return any(ext in url.lower() for ext in extensiones_validas)

    def configurar_api_key(self, api_key: str):
        """Configura la API key de SerpAPI"""
        self.serpapi_key = api_key.strip()
        logger.info(f"API Key configurada: {'✓' if self.serpapi_key != 'YOUR_SERPAPI_KEY' else '✗'}")

    def buscar_y_resumir(self, consulta: str, num_resultados: int = None, num_sentencias: int = None) -> ResultadoBusqueda:
        """Busca información y genera un resumen"""
        if num_resultados is None:
            num_resultados = self.configuracion['num_resultados_busqueda']
        if num_sentencias is None:
            num_sentencias = self.configuracion['num_sentencias_resumen']
        
        texto_completo, enlaces, titulos = self.buscar_contenido_web(consulta, num_resultados)
        
        if not texto_completo:
            return ResultadoBusqueda(
                "No se pudo obtener información sobre este tema.", 
                [], [], [], "error"
            )
        
        resumen = self.resumir_largo(texto_completo, num_sentencias=num_sentencias)
        imagenes = self.buscar_imagenes_google(consulta, num=2)
        
        return ResultadoBusqueda(resumen, enlaces, titulos, imagenes, "busqueda")

    def _seleccionar_participantes(self, tipo_conversacion: TipoConversacion) -> List[str]:
        """Selecciona participantes según el tipo de conversación"""
        participantes_por_tipo = {
            TipoConversacion.DIALOGO: ['estudiante', 'profesor'],
            TipoConversacion.ENTREVISTA: ['periodista', 'experto'],
            TipoConversacion.DEBATE: ['experto', 'historiador'],
            TipoConversacion.TUTORIAL: ['profesor', 'estudiante'],
            TipoConversacion.EXPLICACION: ['robot', 'niño'],
            TipoConversacion.ANALISIS_VIDEO: ['youtuber', 'critico']
        }
        
        roles = participantes_por_tipo.get(tipo_conversacion, ['profesor', 'estudiante'])
        return [self.configuracion['personajes_conversacion'][rol] for rol in roles]

    def crear_conversacion_con_audio(self, tema: str, tipo_conversacion: TipoConversacion = TipoConversacion.DIALOGO, language: str = 'es') -> Tuple[str, List[str]]:
        """Crea una conversación con audio sobre un tema"""
        logger.info(f"Creando conversación tipo: {tipo_conversacion.value} en {language}")
        
        resultado = self.buscar_y_resumir(tema, num_sentencias=10)
        if not resultado.texto or resultado.texto.startswith("No se pudo obtener"):
            return "No se pudo obtener suficiente información para crear la conversación.", []
        
        participantes = self._seleccionar_participantes(tipo_conversacion)
        
        oraciones = resultado.texto.split('. ')
        intercambios = []
        audios = []
        
        for i, oracion in enumerate(oraciones[:10]):
            if oracion.strip():
                participante = participantes[i % len(participantes)]
                texto_adaptado = self._adaptar_texto_participante(oracion, participante)
                intercambios.append((participante, texto_adaptado))
                
                archivo_audio = self.audio_manager.texto_a_audio_avanzado(
                    texto_adaptado, participante.lower(), language
                )
                if archivo_audio:
                    audios.append(archivo_audio)
        
        conversacion_texto = "\n".join([f"**{p}:** {t}" for p, t in intercambios])
        
        return conversacion_texto, audios

    def _adaptar_texto_participante(self, texto: str, participante: str) -> str:
        """Adapta el texto según el tipo de participante"""
        adaptaciones = {
            'Profesor Sabio': f"Como educador, puedo explicar que {texto.lower()}",
            'Estudiante Curioso': f"Me parece interesante que {texto.lower()}",
            'Experto Científico': f"Desde mi experiencia, {texto.lower()}",
            'Niño Preguntón': f"¡Qué cool! {texto}",
            'Robot Inteligente': f"Según mis datos, {texto.lower()}",
            'Historiador': f"Históricamente hablando, {texto.lower()}",
            'Periodista Investigador': f"Según mis investigaciones, {texto.lower()}",
            'YouTuber Entusiasta': f"¡Hola a todos! Hoy les explico que {texto.lower()}",
            'Crítico Analítico': f"Analizando críticamente, observo que {texto.lower()}"
        }
        
        return adaptaciones.get(participante, texto)

    def crear_audiolibro(self, texto: str, language: str = 'es') -> List[str]:
        """Convierte texto en audiolibro"""
        logger.info(f"Creando audiolibro en {language}")
        
        chunk_size = self.configuracion['chunk_size_audio']
        fragmentos = [texto[i:i+chunk_size] for i in range(0, len(texto), chunk_size)]
        
        audios = []
        for i, fragmento in enumerate(fragmentos):
            if fragmento.strip():
                archivo_audio = self.audio_manager.texto_a_audio_avanzado(
                    fragmento, 'narrador', language
                )
                if archivo_audio:
                    audios.append(archivo_audio)
        
        return audios

    def procesar_pdf(self, archivo_pdf: str, num_sentencias: int = 5) -> str:
        """Procesa y resume un archivo PDF"""
        try:
            if not os.path.exists(archivo_pdf):
                return f"Error: El archivo {archivo_pdf} no existe."
            
            import fitz
            doc = fitz.open(archivo_pdf)
            texto_completo = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                texto_completo += page.get_text()
                
                if len(texto_completo) > self.configuracion['max_contenido_pdf']:
                    texto_completo = texto_completo[:self.configuracion['max_contenido_pdf']]
                    break
            
            doc.close()
            
            if not texto_completo.strip():
                return "Error: No se pudo extraer texto del PDF."
            
            return self.resumir_largo(texto_completo, num_sentencias)
            
        except Exception as e:
            logger.error(f"Error procesando PDF: {e}")
            return f"Error al procesar el PDF: {str(e)}"

    def procesar_consulta(self, consulta: str, generar_audio: bool = False) -> Dict[str, Any]:
        """Procesa una consulta del usuario con soporte completo"""
        try:
            idioma = self.detector_idioma.detectar(consulta)
            if idioma not in self.configuracion['idiomas_soportados']:
                idioma = 'es'
            
            tipo_consulta, contenido = self.detectar_tipo_consulta(consulta)
            
            resultado = None
            audios = []
            texto = ""
            metadata = {}
            
            if tipo_consulta == TipoConsulta.YOUTUBE:
                # Procesar video de YouTube
                resultado_youtube = self.procesar_video_youtube(contenido)
                texto = resultado_youtube['texto']
                metadata = resultado_youtube.get('metadata', {})
                
                # Crear conversación sobre el video si se solicita audio
                if generar_audio and resultado_youtube.get('transcripcion'):
                    conversacion, audios = self.crear_conversacion_sobre_video(contenido)
                    texto += f"\n\n**🎭 Conversación generada:**\n{conversacion}"
            
            elif tipo_consulta == TipoConsulta.URL_CHATBOT:
                # Procesar URL de chatbot
                parsed_url = urlparse(contenido)
                titulo = f"Conversación en {parsed_url.netloc}"
                texto = f"**Análisis de URL de Chatbot:**\n\n**URL:** {contenido}\n\n**Información:** Esta es una conversación de chatbot en {parsed_url.netloc}. El contenido específico no puede ser extraído directamente debido a las políticas de privacidad y seguridad.\n\n*Nota: Para obtener el contenido específico de conversaciones de chatbots, es necesario copiar y pegar el texto directamente.*"
            
            elif tipo_consulta == TipoConsulta.COMPARACION:
                resultado = self.buscar_y_resumir(contenido)
                texto = f"**📊 Comparación sobre {contenido}:**\n\n{resultado.texto}\n\n**📚 Fuentes:**\n" + "\n".join([f"- [{t}]({e})" for t, e in zip(resultado.titulos, resultado.enlaces)])
            
            elif tipo_consulta == TipoConsulta.PASO_A_PASO:
                resultado = self.buscar_y_resumir(contenido)
                pasos = [f"{i+1}. {oracion.strip()}" for i, oracion in enumerate(resultado.texto.split('. ')) if oracion.strip()]
                texto = f"**📋 Guía paso a paso: {contenido}**\n\n" + "\n\n".join(pasos[:10]) + "\n\n**📚 Fuentes:**\n" + "\n".join([f"- [{t}]({e})" for t, e in zip(resultado.titulos, resultado.enlaces)])
            
            elif tipo_consulta == TipoConsulta.CONVERSACION:
                texto, audios = self.crear_conversacion_con_audio(contenido, TipoConversacion.DIALOGO, idioma)
                texto = f"**💬 Conversación sobre: {contenido}**\n\n{texto}"
            
            elif tipo_consulta == TipoConsulta.AUDIOLIBRO:
                resultado = self.buscar_y_resumir(contenido)
                audios = self.crear_audiolibro(resultado.texto, idioma)
                texto = f"**🎧 Audiolibro generado: {contenido}**\n\n{resultado.texto}"
            
            elif tipo_consulta == TipoConsulta.PDF:
                texto = self.procesar_pdf(contenido)
                texto = f"**📄 Resumen del PDF:**\n\n{texto}"
            
            else:
                # Consulta general (puede incluir URLs normales)
                if contenido.startswith('http'):
                    # Es una URL normal
                    resultado_url = self._procesar_url_mejorado(contenido)
                    if resultado_url:
                        url_procesada, titulo, contenido_web = resultado_url
                        resumen = self.resumir_texto(contenido_web)
                        texto = f"**🌐 Resumen de la página web:**\n\n**Título:** {titulo}\n\n**URL:** {url_procesada}\n\n**Contenido:**\n{resumen}"
                    else:
                        texto = f"No se pudo procesar la URL: {contenido}"
                else:
                    # Búsqueda general
                    resultado = self.buscar_y_resumir(contenido)
                    texto = f"**🔍 Información sobre: {contenido}**\n\n{resultado.texto}"
                    
                    if resultado.enlaces:
                        texto += "\n\n**📚 Fuentes:**\n" + "\n".join([f"- [{t}]({e})" for t, e in zip(resultado.titulos, resultado.enlaces)])
                    
                    if resultado.imagenes:
                        texto += "\n\n**🖼️ Imágenes relacionadas:**\n" + "\n".join([f"![Imagen]({img})" for img in resultado.imagenes])
            
            # Generar audio si se solicita y no se ha generado ya
            if generar_audio and not audios and texto:
                audios = self.crear_audiolibro(texto, idioma)
            
            # Guardar en historial
            self.historial_global.append((consulta, texto))
            
            return {
                'texto': texto,
                'audios': audios,
                'idioma': idioma,
                'tipo_consulta': tipo_consulta.value,
                'metadata': metadata,
                'exito': True
            }
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return {
                'texto': f"Error procesando la consulta: {str(e)}",
                'audios': [],
                'idioma': 'es',
                'tipo_consulta': 'error',
                'metadata': {},
                'exito': False
            }

    def registrar_feedback(self, consulta: str, es_util: bool):
        """Registra feedback del usuario"""
        self.feedback[consulta] = "útil" if es_util else "no útil"
        logger.info(f"Feedback registrado para consulta '{consulta}': {self.feedback[consulta]}")

    def _limpiar_recursos_automatico(self):
        """Limpia recursos automáticamente al finalizar"""
        try:
            self.temp_manager.cleanup_all()
            self.cache_busquedas._limpiar_cache()
            logger.info("Recursos limpiados automáticamente")
        except Exception as e:
            logger.error(f"Error limpiando recursos: {e}")

# Configuración de logging
def configurar_logging_avanzado():
    import logging
    import logging.handlers
    
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, os.environ.get('CHAT_LOG_LEVEL', 'INFO')))
    
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    handler = logging.handlers.RotatingFileHandler(
        'logs/chat_resumidor.log', 
        maxBytes=1048576, 
        backupCount=5,
        encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def verificar_e_instalar_dependencias():
    import subprocess
    import sys
    
    dependencias = [
        'googlesearch-python', 'requests', 'beautifulsoup4', 'sumy',
        'ipywidgets', 'IPython', 'tqdm', 'google-search-results',
        'googletrans==4.0.0-rc1', 'deep-translator', 'pymupdf',
        'pyttsx3', 'gTTS', 'pygame', 'langdetect', 'youtube-transcript-api',
        'pytube', 'gradio', 'flask', 'flask-socketio'
    ]
    
    for dep in dependencias:
        try:
            module_name = dep.split('==')[0].replace('-', '_')
            if module_name == 'googlesearch_python':
                module_name = 'googlesearch'
            elif module_name == 'google_search_results':
                module_name = 'serpapi'
            elif module_name == 'flask_socketio':
                module_name = 'flask_socketio'
            elif module_name == 'youtube_transcript_api':
                module_name = 'youtube_transcript_api'
            
            __import__(module_name)
        except ImportError:
            print(f"Instalando {dep}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])

def configurar_nltk():
    recursos = ['punkt', 'punkt_tab', 'stopwords']
    for recurso in recursos:
        try:
            if recurso == 'stopwords':
                nltk.data.find(f'corpora/{recurso}')
            else:
                nltk.data.find(f'tokenizers/{recurso}')
        except LookupError:
            print(f"Descargando {recurso}...")
            nltk.download(recurso, quiet=True)

# Inicialización
logger = configurar_logging_avanzado()

try:
    verificar_e_instalar_dependencias()
except Exception as e:
    logger.error(f"Error instalando dependencias: {e}")

configurar_nltk()

# Interfaz mejorada para Jupyter
def crear_interfaz_completa():
    """Crea la interfaz completa con todas las funcionalidades"""
    chat = ChatResumidorAvanzado()
    
    # Widgets de entrada
    entrada_consulta = widgets.Textarea(
        placeholder='Ingresa tu consulta aquí. Ejemplos:\n- "https://youtube.com/watch?v=abc123" (Video de YouTube)\n- "https://chat.openai.com/c/ejemplo" (URL de ChatGPT)\n- "Cómo hacer un pastel paso a paso"\n- "Conversación sobre inteligencia artificial"\n- "Comparar Python vs JavaScript"',
        description='Consulta:',
        layout={'width': '90%', 'height': '140px'}
    )
    
    # Configuración
    api_key_input = widgets.Password(
        placeholder='Ingresa tu API Key de SerpAPI (opcional para mejores imágenes)',
        description='API Key:',
        layout={'width': '70%'}
    )
    
    checkbox_audio = widgets.Checkbox(
        value=False,
        description='Generar audio',
        indent=False
    )
    
    # Nuevas opciones
    web_chat_checkbox = widgets.Checkbox(
        value=False,
        description='Abrir chat web',
        indent=False
    )
    
    gradio_checkbox = widgets.Checkbox(
        value=False,
        description='Usar Gradio',
        indent=False
    )
    
    # Botones
    boton_enviar = widgets.Button(
        description='🚀 Procesar',
        button_style='primary',
        layout={'width': '120px'}
    )
    
    boton_limpiar = widgets.Button(
        description='🧹 Limpiar',
        button_style='warning',
        layout={'width': '120px'}
    )
    
    boton_config = widgets.Button(
        description='⚙️ Config API',
        button_style='info',
        layout={'width': '120px'}
    )
    
    boton_web_chat = widgets.Button(
        description='🌐 Chat Web',
        button_style='success',
        layout={'width': '120px'}
    )
    
    boton_gradio = widgets.Button(
        description='🎨 Gradio',
        button_style='info',
        layout={'width': '120px'}
    )
    
    # Feedback
    boton_util = widgets.Button(
        description='👍 Útil',
        button_style='success',
        layout={'width': '100px'}
    )
    
    boton_no_util = widgets.Button(
        description='👎 No útil',
        button_style='danger',
        layout={'width': '100px'}
    )
    
    # Output
    salida_respuesta = widgets.Output()
    
    # Variables de estado
    ultima_consulta = {'valor': ''}
    
    def on_config_api(b):
        """Configura la API key"""
        with salida_respuesta:
            api_key = api_key_input.value.strip()
            if api_key:
                chat.configurar_api_key(api_key)
                display(Markdown("**✅ API Key configurada correctamente**"))
            else:
                display(Markdown("**⚠️ Por favor, ingresa una API Key válida**"))
    
    def on_web_chat(b):
        """Inicia el servidor web del chat"""
        with salida_respuesta:
            display(Markdown("**🌐 Iniciando servidor web del chat...**"))
            chat.iniciar_servidor_web()
    
    def on_gradio(b):
        """Inicia la interfaz Gradio"""
        with salida_respuesta:
            display(Markdown("**🎨 Iniciando interfaz Gradio...**"))
            chat.iniciar_interfaz_gradio(share=False)
    
    def on_enviar(b):
        """Procesa la consulta del usuario"""
        with salida_respuesta:
            clear_output(wait=True)
            
            consulta = entrada_consulta.value.strip()
            if not consulta:
                display(Markdown("**❌ Error:** Por favor, ingrese una consulta válida."))
                return
            
            ultima_consulta['valor'] = consulta
            
            # Mostrar indicador de carga
            display(Markdown("**🔄 Procesando consulta...** Por favor espera."))
            
            # Procesar consulta
            resultado = chat.procesar_consulta(consulta, generar_audio=checkbox_audio.value)
            
            clear_output(wait=True)
            
            if resultado['exito']:
                # Mostrar tipo de consulta detectado
                tipo_emoji = {
                    'youtube': '📺',
                    'url_chatbot': '🤖',
                    'comparacion': '📊',
                    'paso_a_paso': '📋',
                    'conversacion': '💬',
                    'audiolibro': '🎧',
                    'pdf': '📄',
                    'general': '🔍'
                }
                
                emoji = tipo_emoji.get(resultado['tipo_consulta'], '📝')
                
                display(Markdown(f"""
                **{emoji} Consulta:** {consulta}
                
                **📊 Tipo:** {resultado['tipo_consulta'].title()}
                
                **🌐 Idioma:** {resultado['idioma'].upper()}
                
                ---
                
                {resultado['texto']}
                """))
                
                # Mostrar metadatos adicionales si existen
                if resultado.get('metadata'):
                    metadata = resultado['metadata']
                    if metadata and not metadata.get('error'):
                        display(Markdown(f"""
                        ---
                        **📊 Información adicional:**
                        - **Vistas:** {metadata.get('vistas', 'N/A'):,}
                        - **Duración:** {metadata.get('duracion', 'N/A')}
                        - **Fecha:** {metadata.get('fecha_publicacion', 'N/A')}
                        """))
                
                # Mostrar audios si existen
                if resultado['audios']:
                    from IPython.display import Audio
                    display(Markdown("**🔊 Audio generado:**"))
                    for i, audio in enumerate(resultado['audios']):
                        try:
                            display(Audio(audio, autoplay=False))
                            time.sleep(0.3)
                        except Exception as e:
                            display(Markdown(f"*Error reproduciendo audio {i+1}: {e}*"))
                
                # Mostrar botones de feedback
                display(widgets.HBox([boton_util, boton_no_util]))
                
            else:
                display(Markdown(f"""
                **❌ Error procesando la consulta**
                
                {resultado['texto']}
                """))
    
    def on_util(b):
        """Registra feedback positivo"""
        with salida_respuesta:
            if ultima_consulta['valor']:
                chat.registrar_feedback(ultima_consulta['valor'], True)
                display(Markdown("**✅ ¡Gracias por tu feedback positivo!**"))
            else:
                display(Markdown("**⚠️ No hay consulta para evaluar**"))
    
    def on_no_util(b):
        """Registra feedback negativo"""
        with salida_respuesta:
            if ultima_consulta['valor']:
                chat.registrar_feedback(ultima_consulta['valor'], False)
                display(Markdown("**📝 Gracias por tu feedback. Trabajaremos para mejorar.**"))
            else:
                display(Markdown("**⚠️ No hay consulta para evaluar**"))
    
    def on_limpiar(b):
        """Limpia la interfaz"""
        with salida_respuesta:
            clear_output()
            entrada_consulta.value = ''
            checkbox_audio.value = False
            web_chat_checkbox.value = False
            gradio_checkbox.value = False
            ultima_consulta['valor'] = ''
            display(Markdown("""
            **🧹 Interfaz limpiada**
            
            Ingresa una nueva consulta para continuar. 
            
            **Tipos de consulta soportados:**
            - 🔍 Búsquedas generales
            - 📺 Videos de YouTube (análisis completo con transcripción)
            - 🤖 URLs de chatbots (ChatGPT, Claude, etc.)
            - 📊 Comparaciones (ej: "comparar X vs Y")
            - 📋 Guías paso a paso (ej: "cómo hacer X")
            - 💬 Conversaciones (ej: "conversación sobre X")
            - 🔗 URLs generales
            - 🎧 Audiolibros (ej: "audiolibro de X")
            - 📄 PDFs (ruta del archivo)
            """))
    
    # Conectar eventos
    boton_enviar.on_click(on_enviar)
    boton_util.on_click(on_util)
    boton_no_util.on_click(on_no_util)
    boton_limpiar.on_click(on_limpiar)
    boton_config.on_click(on_config_api)
    boton_web_chat.on_click(on_web_chat)
    boton_gradio.on_click(on_gradio)
    
    # Mostrar interfaz
    from IPython.display import display, Markdown, clear_output
    
    display(Markdown("""
    # 🤖 Chat Resumidor Avanzado - Versión Completa
    
    **Nuevas características:**
    - ✅ **📺 Análisis completo de videos de YouTube** (transcripción + metadatos)
    - ✅ **🌐 Chat web interactivo** con servidor Flask + SocketIO
    - ✅ **🎨 Interfaz Gradio** para uso web alternativo
    - ✅ **🤖 Soporte para URLs de chatbots** (ChatGPT, Claude, etc.)
    - ✅ **🔍 Búsquedas inteligentes** con resúmenes mejorados
    - ✅ **🎧 Generación de audio** y conversaciones
    - ✅ **📄 Procesamiento de PDFs**
    - ✅ **💾 Cache inteligente** y sistema de feedback
    """))
    
    # Configuración
    display(Markdown("### ⚙️ Configuración"))
    display(widgets.HBox([api_key_input, boton_config]))
    
    # Entrada principal
    display(Markdown("### 📝 Consulta"))
    display(entrada_consulta)
    
    # Opciones
    display(Markdown("### 🔧 Opciones"))
    display(widgets.HBox([checkbox_audio, web_chat_checkbox, gradio_checkbox]))
    
    # Botones principales
    display(Markdown("### 🎮 Acciones"))
    display(widgets.HBox([boton_enviar, boton_limpiar, boton_web_chat, boton_gradio]))
    
    # Output
    display(salida_respuesta)
    
    # Mensaje inicial
    with salida_respuesta:
        display(Markdown("""
        **🚀 ¡Bienvenido al Chat Resumidor Avanzado!**
        
        **Ejemplos de uso:**
        
        **📺 YouTube:** `https://youtube.com/watch?v=abc123`
        - Extrae transcripción, metadatos y genera resumen completo
        
        **🤖 Chatbots:** `https://chat.openai.com/c/ejemplo`
        - Analiza URLs de conversaciones de chatbots
        
        **🔍 Búsquedas:** `Explícame sobre inteligencia artificial`
        - Busca información actualizada y genera resúmenes
        
        **📊 Comparaciones:** `Comparar Python vs JavaScript`
        - Análisis detallado de diferencias y similitudes
        
        **💬 Conversaciones:** `Conversación sobre cambio climático`
        - Genera diálogos educativos con audio
        
        **🎧 Audiolibros:** `Audiolibro sobre historia de España`
        - Convierte contenido a formato de audio narrado
        
        **💡 Consejo:** Para mejores resultados con imágenes, configura tu API Key de SerpAPI.
        
        **🌐 Chat Web:** Haz clic en "Chat Web" para abrir una interfaz web completa.
        
        **🎨 Gradio:** Usa "Gradio" para una interfaz web alternativa moderna.
        """))

    return chat

# Función auxiliar para uso directo
def crear_chat_simple():
    """Crea una instancia simple del chat para uso programático"""
    return ChatResumidorAvanzado()

# Función para iniciar solo el servidor web
def iniciar_solo_servidor_web(host='localhost', port=5000, api_key=None):
    """Inicia solo el servidor web sin interfaz Jupyter"""
    chat = ChatResumidorAvanzado(api_key)
    print("🌐 Iniciando servidor web...")
    chat.iniciar_servidor_web(host, port)
    
    # Mantener el servidor corriendo
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario")

# Función para iniciar solo Gradio
def iniciar_solo_gradio(share=False, api_key=None):
    """Inicia solo la interfaz Gradio"""
    chat = ChatResumidorAvanzado(api_key)
    print("🎨 Iniciando interfaz Gradio...")
    chat.iniciar_interfaz_gradio(share)

# Función de ejemplo para usar programáticamente
def ejemplo_uso_programatico():
    """Ejemplo de cómo usar el chat programáticamente"""
    chat = ChatResumidorAvanzado()
    
    # Configurar API key si tienes una
    # chat.configurar_api_key("tu_api_key_aqui")
    
    # Ejemplos de uso
    ejemplos = [
        "https://youtube.com/watch?v=dQw4w9WgXcQ",  # Video de YouTube
        "Cómo hacer pan casero paso a paso",
        "Conversación sobre inteligencia artificial",
        "Comparar energía solar vs energía eólica",
        "https://chat.openai.com/c/ejemplo"  # URL de ChatGPT
    ]
    
    for consulta in ejemplos:
        print(f"\n{'='*50}")
        print(f"Procesando: {consulta}")
        print('='*50)
        
        resultado = chat.procesar_consulta(consulta, generar_audio=False)
        
        if resultado['exito']:
            print(f"Tipo: {resultado['tipo_consulta']}")
            print(f"Idioma: {resultado['idioma']}")
            print(f"Texto: {resultado['texto'][:500]}...")
            if resultado['audios']:
                print(f"Audios generados: {len(resultado['audios'])}")
        else:
            print(f"Error: {resultado['texto']}")

# Ejecutar la interfaz principal
if __name__ == "__main__":
    # Verificar si estamos en Jupyter
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            # Estamos en Jupyter, mostrar interfaz completa
            crear_interfaz_completa()
        else:
            # Estamos en script normal, mostrar opciones
            print("🤖 Chat Resumidor Avanzado")
            print("Opciones disponibles:")
            print("1. crear_interfaz_completa() - Interfaz Jupyter completa")
            print("2. iniciar_solo_servidor_web() - Solo servidor web")
            print("3. iniciar_solo_gradio() - Solo interfaz Gradio")
            print("4. ejemplo_uso_programatico() - Ejemplo de uso en código")
    except ImportError:
        # No está IPython disponible
        print("🤖 Chat Resumidor Avanzado")
        print("Para usar en Jupyter: crear_interfaz_completa()")
        print("Para servidor web: iniciar_solo_servidor_web()")
        print("Para Gradio: iniciar_solo_gradio()")
else:
    print("📚 Módulo cargado correctamente.")
    print("🚀 Ejecuta crear_interfaz_completa() para iniciar la interfaz completa.")
    print("🌐 O usa iniciar_solo_servidor_web() para el chat web.")
    print("🎨 O usa iniciar_solo_gradio() para la interfaz Gradio.")
    
    # Auto-iniciar la interfaz si estamos en Jupyter
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            crear_interfaz_completa()
    except:
        pass# Instalación de dependencias mejoradas
%pip install googlesearch-python requests beautifulsoup4 sumy ipywidgets IPython
%pip install tqdm google-search-results googletrans==4.0.0-rc1 deep-translator pymupdf pyttsx3 gTTS pygame langdetect
%pip install youtube-transcript-api pytube gradio flask-socketio websockets

import nltk
import os
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import logging.handlers
import pyttsx3
from gtts import gTTS
import pygame
import io
import tempfile
import time
import warnings
import re
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from urllib.parse import urlparse, urljoin, parse_qs
from requests.exceptions import Timeout, ConnectionError, RequestException
import hashlib
import threading
import queue
import webbrowser

# Configuración inicial
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Nuevas importaciones para YouTube y Web Chat
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from pytube import YouTube
    import gradio as gr
    from flask import Flask, render_template
    from flask_socketio import SocketIO, emit, send
    import websockets
    YOUTUBE_AVAILABLE = True
    WEB_CHAT_AVAILABLE = True
except ImportError as e:
    print(f"Algunas funcionalidades no estarán disponibles: {e}")
    YOUTUBE_AVAILABLE = False
    WEB_CHAT_AVAILABLE = False

# Clases de soporte mejoradas
class TempFileManager:
    def __init__(self):
        self.temp_files = []
        self.lock = threading.Lock()

    def create_temp_file(self, suffix='.mp3'):
        with self.lock:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            self.temp_files.append(temp_file.name)
            return temp_file.name

    def cleanup_all(self):
        with self.lock:
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    print(f"Error al eliminar archivo temporal {temp_file}: {e}")
            self.temp_files.clear()

class YouTubeProcessor:
    """Procesador especializado para videos de YouTube"""
    
    def __init__(self, cache):
        self.cache = cache
        
    def extraer_video_id(self, url: str) -> Optional[str]:
        """Extrae el ID del video de YouTube de una URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def obtener_transcripcion(self, video_id: str, idiomas: List[str] = ['es', 'en']) -> Optional[str]:
        """Obtiene la transcripción de un video de YouTube"""
        if not YOUTUBE_AVAILABLE:
            return None
            
        cache_key = f"youtube_transcript_{video_id}"
        resultado_cache = self.cache.get(cache_key)
        if resultado_cache:
            return resultado_cache
        
        try:
            # Intentar obtener transcripción en los idiomas especificados
            for idioma in idiomas:
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[idioma])
                    texto_completo = ' '.join([entry['text'] for entry in transcript])
                    self.cache.set(cache_key, texto_completo)
                    return texto_completo
                except:
                    continue
            
            # Si no hay transcripción en los idiomas preferidos, usar cualquiera disponible
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript in transcript_list:
                try:
                    transcript_data = transcript.fetch()
                    texto_completo = ' '.join([entry['text'] for entry in transcript_data])
                    self.cache.set(cache_key, texto_completo)
                    return texto_completo
                except:
                    continue
                    
        except Exception as e:
            print(f"Error obteniendo transcripción de YouTube: {e}")
        
        return None
    
    def obtener_metadata_video(self, video_id: str) -> Dict[str, Any]:
        """Obtiene metadatos del video de YouTube"""
        if not YOUTUBE_AVAILABLE:
            return {}
            
        cache_key = f"youtube_metadata_{video_id}"
        resultado_cache = self.cache.get(cache_key)
        if resultado_cache:
            return resultado_cache
        
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            yt = YouTube(url)
            
            metadata = {
                'titulo': yt.title,
                'autor': yt.author,
                'duracion': str(yt.length) + ' segundos',
                'vistas': yt.views,
                'descripcion': yt.description[:500] + '...' if len(yt.description) > 500 else yt.description,
                'fecha_publicacion': str(yt.publish_date),
                'url': url
            }
            
            self.cache.set(cache_key, metadata)
            return metadata
            
        except Exception as e:
            print(f"Error obteniendo metadatos del video: {e}")
            return {'error': str(e)}
    
    def procesar_video_youtube(self, url: str) -> Optional[Dict[str, Any]]:
        """Procesa completamente un video de YouTube"""
        video_id = self.extraer_video_id(url)
        if not video_id:
            return None
        
        # Obtener metadatos
        metadata = self.obtener_metadata_video(video_id)
        
        # Obtener transcripción
        transcripcion = self.obtener_transcripcion(video_id)
        
        if not transcripcion and not metadata:
            return None
        
        return {
            'video_id': video_id,
            'url': url,
            'metadata': metadata,
            'transcripcion': transcripcion,
            'tipo': 'youtube'
        }

class WebChatServer:
    """Servidor web para chat interactivo"""
    
    def __init__(self, chat_resumidor):
        self.chat_resumidor = chat_resumidor
        self.app = None
        self.socketio = None
        self.server_thread = None
        self.active_connections = set()
        
    def crear_app_flask(self):
        """Crea la aplicación Flask con SocketIO"""
        if not WEB_CHAT_AVAILABLE:
            return None
            
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'chat_resumidor_secret_key'
        socketio = SocketIO(app, cors_allowed_origins="*")
        
        @app.route('/')
        def index():
            return render_template('chat.html')
        
        @socketio.on('connect')
        def handle_connect():
            self.active_connections.add(request.sid)
            emit('status', {'msg': 'Conectado al Chat Resumidor Avanzado'})
        
        @socketio.on('disconnect')
        def handle_disconnect():
            self.active_connections.discard(request.sid)
        
        @socketio.on('consulta')
        def handle_consulta(data):
            consulta = data.get('consulta', '').strip()
            generar_audio = data.get('audio', False)
            
            if not consulta:
                emit('error', {'msg': 'Consulta vacía'})
                return
            
            try:
                # Procesar consulta
                resultado = self.chat_resumidor.procesar_consulta(consulta, generar_audio)
                emit('respuesta', resultado)
            except Exception as e:
                emit('error', {'msg': f'Error procesando consulta: {str(e)}'})
        
        @socketio.on('feedback')
        def handle_feedback(data):
            consulta = data.get('consulta', '')
            es_util = data.get('util', True)
            self.chat_resumidor.registrar_feedback(consulta, es_util)
            emit('status', {'msg': 'Feedback registrado'})
        
        return app, socketio
    
    def crear_template_html(self):
        """Crea el template HTML para el chat web"""
        template_dir = os.path.join(os.getcwd(), 'templates')
        os.makedirs(template_dir, exist_ok=True)
        
        html_content = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Resumidor Avanzado</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 800px;
            height: 80vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 10px;
            animation: fadeIn 0.3s ease-in;
        }
        .user-message {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            margin-left: 20%;
        }
        .bot-message {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
            margin-right: 20%;
        }
        .error-message {
            background: #ffebee;
            border-left: 4px solid #f44336;
            color: #c62828;
        }
        .status-message {
            background: #e8f5e8;
            border-left: 4px solid #4caf50;
            color: #2e7d32;
            font-style: italic;
        }
        .chat-input-container {
            display: flex;
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            gap: 10px;
        }
        .chat-input {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        .chat-input:focus {
            border-color: #667eea;
        }
        .send-button, .audio-toggle {
            padding: 15px 20px;
            border: none;
            border-radius: 25px;
            background: #667eea;
            color: white;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        .send-button:hover, .audio-toggle:hover {
            background: #5a6fd8;
        }
        .audio-toggle.active {
            background: #4caf50;
        }
        .feedback-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .feedback-btn {
            padding: 5px 15px;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
        }
        .feedback-btn.positive {
            background: #4caf50;
            color: white;
        }
        .feedback-btn.negative {
            background: #f44336;
            color: white;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            🤖 Chat Resumidor Avanzado
            <div style="font-size: 0.6em; margin-top: 5px;">
                Busca información, analiza YouTube, procesa URLs
            </div>
        </div>
        
        <div class="chat-messages" id="messages">
            <div class="message status-message">
                ¡Bienvenido! Puedes:
                <ul style="margin-top: 10px; margin-left: 20px;">
                    <li>Buscar información general</li>
                    <li>Analizar videos de YouTube</li>
                    <li>Procesar URLs y documentos</li>
                    <li>Crear conversaciones y audiolibros</li>
                </ul>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            Procesando consulta...
        </div>
        
        <div class="chat-input-container">
            <input type="text" id="messageInput" class="chat-input" 
                   placeholder="Escribe tu consulta aquí... (ej: 'https://youtube.com/watch?v=...')">
            <button id="audioToggle" class="audio-toggle" title="Toggle Audio">🔊</button>
            <button id="sendButton" class="send-button">Enviar</button>
        </div>
    </div>

    <script>
        const socket = io();
        const messagesContainer = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const audioToggle = document.getElementById('audioToggle');
        const loading = document.getElementById('loading');
        
        let audioEnabled = false;
        let currentConsulta = '';
        
        // Toggle audio
        audioToggle.addEventListener('click', () => {
            audioEnabled = !audioEnabled;
            audioToggle.classList.toggle('active', audioEnabled);
            audioToggle.textContent = audioEnabled ? '🔊' : '🔇';
        });
        
        // Send message
        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            currentConsulta = message;
            addMessage(message, 'user-message');
            
            loading.style.display = 'block';
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            socket.emit('consulta', {
                consulta: message,
                audio: audioEnabled
            });
            
            messageInput.value = '';
        }
        
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        // Add message to chat
        function addMessage(content, className, showFeedback = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${className}`;
            
            if (className === 'bot-message') {
                messageDiv.innerHTML = content;
                
                if (showFeedback) {
                    const feedbackDiv = document.createElement('div');
                    feedbackDiv.className = 'feedback-buttons';
                    feedbackDiv.innerHTML = `
                        <button class="feedback-btn positive" onclick="sendFeedback(true)">👍 Útil</button>
                        <button class="feedback-btn negative" onclick="sendFeedback(false)">👎 No útil</button>
                    `;
                    messageDiv.appendChild(feedbackDiv);
                }
            } else {
                messageDiv.textContent = content;
            }
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        // Send feedback
        function sendFeedback(isUseful) {
            socket.emit('feedback', {
                consulta: currentConsulta,
                util: isUseful
            });
        }
        
        // Socket events
        socket.on('connect', () => {
            console.log('Conectado al servidor');
        });
        
        socket.on('status', (data) => {
            addMessage(data.msg, 'status-message');
        });
        
        socket.on('error', (data) => {
            loading.style.display = 'none';
            addMessage(`Error: ${data.msg}`, 'error-message');
        });
        
        socket.on('respuesta', (data) => {
            loading.style.display = 'none';
            
            let content = `
                <div style="margin-bottom: 10px;">
                    <strong>📝 Tipo:</strong> ${data.tipo_consulta || 'General'}<br>
                    <strong>🌐 Idioma:</strong> ${(data.idioma || 'es').toUpperCase()}
                </div>
                <div style="border-top: 1px solid #ddd; padding-top: 10px;">
                    ${data.texto.replace(/\n/g, '<br>')}
                </div>
            `;
            
            if (data.audios && data.audios.length > 0) {
                content += '<div style="margin-top: 15px;"><strong>🔊 Audio:</strong>';
                data.audios.forEach((audio, index) => {
                    content += `<br><audio controls style="width: 100%; margin-top: 5px;"><source src="${audio}" type="audio/mpeg">Tu navegador no soporta audio.</audio>`;
                });
                content += '</div>';
            }
            
            addMessage(content, 'bot-message', true);
        });
    </script>
</body>
</html>
        '''
        
        with open(os.path.join(template_dir, 'chat.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def iniciar_servidor(self, host='localhost', port=5000):
        """Inicia el servidor web del chat"""
        if not WEB_CHAT_AVAILABLE:
            print("Flask/SocketIO no está disponible. Instala: pip install flask flask-socketio")
            return
        
        try:
            self.crear_template_html()
            self.app, self.socketio = self.crear_app_flask()
            
            def run_server():
                self.socketio.run(self.app, host=host, port=port, debug=False, use_reloader=False)
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            time.sleep(2)  # Esperar a que el servidor inicie
            url = f"http://{host}:{port}"
            print(f"🌐 Servidor web iniciado en: {url}")
            
            try:
                webbrowser.open(url)
                print("🚀 Abriendo navegador automáticamente...")
            except:
                print("💡 Abre manualmente el navegador y ve a la URL mostrada arriba")
                
        except Exception as e:
            print(f"Error iniciando servidor web: {e}")

class GradioInterface:
    """Interfaz con Gradio para uso web alternativo"""
    
    def __init__(self, chat_resumidor):
        self.chat_resumidor = chat_resumidor
    
    def crear_interfaz_gradio(self):
        """Crea interfaz web con Gradio"""
        if not WEB_CHAT_AVAILABLE:
            return None
        
        def procesar_consulta_gradio(consulta, audio_enabled, api_key):
            if api_key:
                self.chat_resumidor.configurar_api_key(api_key)
            
            resultado = self.chat_resumidor.procesar_consulta(consulta, audio_enabled)
            
            return resultado['texto'], resultado.get('audios', [])
        
        with gr.Blocks(title="Chat Resumidor Avanzado", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Chat Resumidor Avanzado")
            gr.Markdown("Busca información, analiza YouTube, procesa URLs y más...")
            
            with gr.Row():
                with gr.Column(scale=3):
                    consulta_input = gr.Textbox(
                        label="Tu consulta",
                        placeholder="Ejemplo: https://youtube.com/watch?v=... o 'Explícame sobre inteligencia artificial'",
                        lines=3
                    )
                    
                with gr.Column(scale=1):
                    audio_checkbox = gr.Checkbox(label="Generar Audio", value=False)
                    api_key_input = gr.Textbox(
                        label="API Key (SerpAPI)", 
                        type="password",
                        placeholder="Opcional para mejores resultados"
                    )
            
            procesar_btn = gr.Button("🚀 Procesar", variant="primary")
            
            with gr.Row():
                respuesta_output = gr.Textbox(label="Respuesta", lines=15)
                audio_output = gr.Audio(label="Audio Generado", visible=False)
            
            procesar_btn.click(
                procesar_consulta_gradio,
                inputs=[consulta_input, audio_checkbox, api_key_input],
                outputs=[respuesta_output, audio_output]
            )
        
        return interface

class URLValidator:
    @staticmethod
    def es_url_segura(url: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                return False
            if not parsed.netloc:
                return False
            
            # Expandir dominios seguros incluyendo YouTube
            dominios_seguros = [
                'youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com',
                'openai.com', 'chat.openai.com', 'claude.ai', 'bard.google.com',
                'wikipedia.org', 'github.com', 'stackoverflow.com', 'medium.com',
                'arxiv.org', 'scholar.google.com', 'reddit.com'
            ]
            blacklisted_domains = ['malware.com', 'phishing.com', 'spam.com']
            
            if any(domain in parsed.netloc.lower() for domain in dominios_seguros):
                return True
            
            return not any(domain in parsed.netloc.lower() for domain in blacklisted_domains)
        except Exception:
            return False

    @staticmethod
    def es_url_youtube(url: str) -> bool:
        """Detecta si la URL es de YouTube"""
        youtube_patterns = [
            r'(?:youtube\.com|youtu\.be)',
            r'youtube\.com/watch\?',
            r'youtu\.be/',
            r'youtube\.com/embed/'
        ]
        return any(re.search(pattern, url.lower()) for pattern in youtube_patterns)

    @staticmethod
    def es_url_chatbot(url: str) -> bool:
        """Detecta si la URL es de un chatbot como ChatGPT"""
        chatbot_patterns = [
            r'chat\.openai\.com',
            r'claude\.ai',
            r'bard\.google\.com',
            r'copilot\.microsoft\.com',
            r'character\.ai',
            r'poe\.com'
        ]
        return any(re.search(pattern, url.lower()) for pattern in chatbot_patterns)

# Actualizar AudioManagerMejorado con nuevas capacidades
class AudioManagerMejorado:
    def __init__(self):
        try:
            pygame.mixer.init()
            self.engine = pyttsx3.init()
            self.temp_manager = TempFileManager()
            self.configurar_voz()
        except Exception as e:
            print(f"Error inicializando audio: {e}")
            self.engine = None

    def configurar_voz(self):
        if not self.engine:
            return
        try:
            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'spanish' in voice.name.lower() or 'es-' in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            self.engine.setProperty('rate', int(os.environ.get('CHAT_AUDIO_RATE', 180)))
            self.engine.setProperty('volume', float(os.environ.get('CHAT_AUDIO_VOLUME', 0.9)))
        except Exception as e:
            print(f"Error configurando voz: {e}")

    def texto_a_audio_avanzado(self, texto: str, participante: str = "default", language: str = 'es') -> Optional[str]:
        try:
            texto_limpio = re.sub(r'<[^>]+>', '', texto)
            texto_limpio = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', texto_limpio)
            texto_limpio = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', texto_limpio)
            
            if len(texto_limpio.strip()) < 5:
                return None
                
            # Configurar TLD según el participante
            tld_map = {
                'profesor': 'com', 'experto': 'com', 'estudiante': 'com.au',
                'niño': 'com.au', 'robot': 'co.uk', 'narrador': 'com'
            }
            tld = tld_map.get(participante.lower(), 'com')
            
            tts = gTTS(text=texto_limpio, lang=language, tld=tld, slow=False)
            archivo_audio = self.temp_manager.create_temp_file(suffix='.mp3')
            tts.save(archivo_audio)
            return archivo_audio
        except Exception as e:
            print(f"Error generando audio avanzado para {language}: {e}")
            return None

    def reproducir_audio(self, archivo_audio: str):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(archivo_audio)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Error reproduciendo audio: {e}")

# Mantener otras clases existentes (CacheInteligente, DetectorIdioma, etc.)
class CacheInteligente:
    def __init__(self, max_size: int = 200, ttl_hours: int = 48):
        self.cache = {}
        self.max_size = int(os.environ.get('CHAT_CACHE_SIZE', max_size))
        self.ttl = ttl_hours * 3600
        self.timestamps = {}
        self.lock = threading.RLock()

    def _generar_clave(self, key: str) -> str:
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def get(self, key: str):
        with self.lock:
            cache_key = self._generar_clave(key)
            if cache_key in self.cache:
                if time.time() - self.timestamps[cache_key] < self.ttl:
                    return self.cache[cache_key]
                else:
                    del self.cache[cache_key]
                    del self.timestamps[cache_key]
            return None

    def set(self, key: str, value):
        with self.lock:
            cache_key = self._generar_clave(key)
            if len(self.cache) >= self.max_size:
                self._limpiar_cache()
            self.cache[cache_key] = value
            self.timestamps[cache_key] = time.time()

    def _limpiar_cache(self):
        current_time = time.time()
        expired = [k for k, t in self.timestamps.items() if current_time - t >= self.ttl]
        for key in expired:
            if key in self.cache:
                del self.cache[key]
            if key in self.timestamps:
                del self.timestamps[key]

class DetectorIdioma:
    def detectar(self, texto: str) -> str:
        try:
            from langdetect import detect
            return detect(texto)
        except:
            return 'es'

class ValidadorContenido:
    @staticmethod
    def es_contenido_valido(texto: str, url: str) -> bool:
        if len(texto) < 50:
            return False
        spam_keywords = ['login', 'signup', 'advertisement', 'cookies', 'subscribe']
        texto_lower = texto.lower()
        spam_count = sum(1 for keyword in spam_keywords if keyword in texto_lower)
        return spam_count < 3

# Enums y DataClasses actualizados
class TipoConsulta(Enum):
    COMPARACION = "comparacion"
    PASO_A_PASO = "paso_a_paso"
    CONVERSACION = "conversacion"
    GENERAL = "general"
    PDF = "pdf"
    AUDIOLIBRO = "audiolibro"
    RESUMEN_FORMATEADO = "resumen_formateado"
    URL_CHATBOT = "url_chatbot"
    YOUTUBE = "youtube"  # Nuevo tipo
    WEB_CHAT = "web_chat"  # Nuevo tipo

class TipoConversacion(Enum):
    DIALOGO = "dialogo"
    ENTREVISTA = "entrevista"
    DEBATE = "debate"
    TUTORIAL = "tutorial"
    EXPLICACION = "explicacion"
    ANALISIS_VIDEO = "analisis_video"  # Nuevo tipo

@dataclass
class ResultadoBusqueda:
    texto: str
    enlaces: List[str]
    titulos: List[str]
    imagenes: List[str]
    tipo: str = "general"
    metadata: Dict[str, Any] = None

@dataclass
class Conversacion:
    participantes: List[str]
    intercambios: List[Tuple[str, str]]
    tipo: TipoConversacion
    tema: str

# Clase principal mejorada con YouTube y Web Chat
class ChatResumidorAvanzado:
    def __init__(self, serpapi_key: str = None):
        self.serpapi_key = serpapi_key or os.environ.get('SERPAPI_KEY', "YOUR_SERPAPI_KEY")
        self.historial_global = []
        self.feedback = {}
        self.cache_busquedas = CacheInteligente(max_size=300, ttl_hours=48)
        self.audio_manager = AudioManagerMejorado()
        self.detector_idioma = DetectorIdioma()
        self.temp_manager = TempFileManager()
        
        # Nuevos procesadores
        self.youtube_processor = YouTubeProcessor(self.cache_busquedas)
        self.web_chat_server = WebChatServer(self)
        self.gradio_interface = GradioInterface(self)
        
        self.configuracion = {
            'num_resultados_busqueda': 5,
            'num_sentencias_resumen': 8,
            'fragmento_max': 15000,
            'max_iteraciones': 3,
            'timeout_request': 15,
            'max_imagenes': 3,
            'idiomas_soportados': ['es', 'en', 'fr', 'de', 'pt', 'it'],
            'max_contenido_pdf': 50000,
            'chunk_size_audio': 2000,
            'reintentos_url': 2,
            'max_transcripcion_youtube': 10000,
            'personajes_conversacion': {
                'profesor': 'Profesor Sabio',
                'estudiante': 'Estudiante Curioso',
                'experto': 'Experto Científico',
                'niño': 'Niño Preguntón',
                'robot': 'Robot Inteligente',
                'historiador': 'Historiador',
                'artista': 'Artista Creativo',
                'chef': 'Chef Experto',
                'periodista': 'Periodista Investigador',
                'narrador': 'Narrador Profesional',
                'youtuber': 'YouTuber Entusiasta',
                'critico': 'Crítico Analítico'
            }
        }
        
        # Patrones mejorados con soporte para YouTube
        self.patrones = {
            TipoConsulta.COMPARACION: [
                r'compar[ae]r?\s+(.+?)\s+(?:vs|con|versus|y)\s+(.+)',
                r'(?:diferencias?|similitudes?)\s+entre\s+(.+?)\s+y\s+(.+)',
                r'cuadro\s+comparativo\s+(.+)',
                r'(.+?)\s+(?:vs|versus)\s+(.+)'
            ],
            TipoConsulta.PASO_A_PASO: [
                r'(?:como?|cómo)\s+hacer\s+(.+?)(?:\s+paso\s+a\s+paso)?',
                r'(?:tutorial|guía)\s+(?:de|para)\s+(.+)',
                r'pasos?\s+para\s+(.+)',
                r'instrucciones?\s+(?:de|para)\s+(.+)'
            ],
            TipoConsulta.CONVERSACION: [
                r'convierte?\s+(?:a|en)\s+conversaci[oó]n\s+(.+)',
                r'dialogo\s+(?:sobre|de)\s+(.+)',
                r'entrevista\s+(?:sobre|de)\s+(.+)',
                r'conversaci[oó]n\s+(?:sobre|entre|de)\s+(.+)',
                r'debate\s+(?:sobre|entre)\s+(.+)',
                r'explica\s+(?:como|en)\s+conversaci[oó]n\s+(.+)'
            ],
            TipoConsulta.AUDIOLIBRO: [
                r'audiolibro\s+(?:de|del)\s+(.+)',
                r'convierte?\s+(?:a|en)\s+audio\s+(.+)',
                r'lee\s+(?:el|la)\s+(.+)',
                r'narrar?\s+(.+)'
            ],
            TipoConsulta.YOUTUBE: [
                r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#\s]+)',
                r'analiza?\s+(?:este|el)\s+video\s+de\s+youtube[:\s]+(.+)',
                r'resumir?\s+(?:este|el)\s+video[:\s]+(.+)',
                r'transcripci[oó]n\s+de\s+(.+)',
                r'qu[eé]\s+dice\s+(?:este|el)\s+video\s+(.+)'
            ],
            TipoConsulta.URL_CHATBOT: [
                r'https?://chat\.openai\.com/[^\s]+',
                r'https?://claude\.ai/[^\s]+',
                r'analiza?\s+(?:esta|la)\s+conversaci[oó]n[:\s]+(.+)',
                r'resumir?\s+(?:este|el)\s+chat[:\s]+(.+)'
            ]
        }
        
        import atexit
        atexit.register(self._limpiar_recursos_automatico)

    def detectar_tipo_consulta(self, consulta: str) -> Tuple[TipoConsulta, Optional[str]]:
        consulta_stripped = consulta.strip()
        consulta_lower = consulta_stripped.lower()
        
        # Detectar YouTube primero
        if URLValidator.es_url_youtube(consulta_stripped):
            return TipoConsulta.YOUTUBE, consulta_stripped
        
        # Detectar URLs de chatbots
        if URLValidator.es_url_chatbot(consulta_stripped):
            return TipoConsulta.URL_CHATBOT, consulta_stripped
        
        # Detectar URLs generales
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, consulta_stripped)
        if urls:
            if URLValidator.es_url_youtube(urls[0]):
                return TipoConsulta.YOUTUBE, urls[0]
            elif URLValidator.es_url_chatbot(urls[0]):
                return TipoConsulta.URL_CHATBOT, urls[0]
            else:
                return TipoConsulta.GENERAL, urls[0]  # URL general
        
        # Detectar otros tipos de consulta
        for tipo, patrones in self.patrones.items():
            if tipo in [TipoConsulta.YOUTUBE, TipoConsulta.URL_CHATBOT]:
                continue  # Ya procesamos arriba
                
            for patron in patrones:
                match = re.search(patron, consulta_lower)
                if match:
                    contenido = match.group(1) if match.groups() else consulta_stripped
                    return tipo, contenido.strip()
        
        return TipoConsulta.GENERAL, consulta_stripped

    def procesar_video_youtube(self, url: str) -> Dict[str, Any]:
        """Procesa un video de YouTube y extrae información"""
        resultado_video = self.youtube_processor.procesar_video_youtube(url)
        
        if not resultado_video:
            return {
                'texto': f"No se pudo procesar el video de YouTube: {url}",
                'tipo': 'error'
            }
        
        # Formatear resultado
        metadata = resultado_video.get('metadata', {})
        transcripcion = resultado_video.get('transcripcion', '')
        
        texto_resultado = f"**📺 Análisis de Video de YouTube**\n\n"
        
        if metadata and not metadata.get('error'):
            texto_resultado += f"**🎬 Título:** {metadata.get('titulo', 'N/A')}\n\n"
            texto_resultado += f"**👤 Canal:** {metadata.get('autor', 'N/A')}\n\n"
            texto_resultado += f"**⏱️ Duración:** {metadata.get('duracion', 'N/A')}\n\n"
            texto_resultado += f"**👁️ Vistas:** {metadata.get('vistas', 'N/A'):,}\n\n"
            
            if metadata.get('fecha_publicacion'):
                texto_resultado += f"**📅 Publicado:** {metadata.get('fecha_publicacion')}\n\n"
        
        if transcripcion:
            # Resumir la transcripción si es muy larga
            if len(transcripcion) > self.configuracion['max_transcripcion_youtube']:
                transcripcion = transcripcion[:self.configuracion['max_transcripcion_youtube']] + "..."
            
            resumen_transcripcion = self.resumir_texto(transcripcion, num_sentencias=6)
            texto_resultado += f"**📝 Resumen del contenido:**\n{resumen_transcripcion}\n\n"
            
            # Generar puntos clave
            oraciones = transcripcion.split('. ')
            puntos_clave = [oracion.strip() for oracion in oraciones[:5] if len(oracion.strip()) > 30]
            if puntos_clave:
                texto_resultado += "**🔑 Puntos clave mencionados:**\n"
                for i, punto in enumerate(puntos_clave, 1):
                    texto_resultado += f"{i}. {punto}\n"
                texto_resultado += "\n"
        else:
            texto_resultado += "**⚠️ Nota:** No se pudo obtener la transcripción del video. Esto puede deberse a que:\n"
            texto_resultado += "- El video no tiene subtítulos automáticos activados\n"
            texto_resultado += "- Los subtítulos están en un idioma no soportado\n"
            texto_resultado += "- El video es privado o tiene restricciones\n\n"
        
        texto_resultado += f"**🔗 URL:** {url}"
        
        return {
            'texto': texto_resultado,
            'metadata': metadata,
            'transcripcion': transcripcion,
            'tipo': 'youtube'
        }

    def crear_conversacion_sobre_video(self, url: str, tipo_conversacion: TipoConversacion = TipoConversacion.ANALISIS_VIDEO) -> Tuple[str, List[str]]:
        """Crea una conversación sobre un video de YouTube"""
        resultado_video = self.youtube_processor.procesar_video_youtube(url)
        
        if not resultado_video or not resultado_video.get('transcripcion'):
            return "No se pudo crear la conversación. Video sin transcripción disponible.", []
        
        metadata = resultado_video.get('metadata', {})
        transcripcion = resultado_video.get('transcripcion', '')
        
        # Seleccionar participantes según el tipo
        if tipo_conversacion == TipoConversacion.ANALISIS_VIDEO:
            participantes = ['YouTuber Entusiasta', 'Crítico Analítico']
        else:
            participantes = self._seleccionar_participantes(tipo_conversacion)
        
        # Crear intercambios basados en el contenido del video
        titulo = metadata.get('titulo', 'Este video')
        autor = metadata.get('autor', 'el creador')
        
        intercambios = [
            (participantes[0], f"Hoy vamos a analizar el video '{titulo}' de {autor}. ¿Qué te parece el contenido?"),
        ]
        
        # Dividir transcripción en segmentos para la conversación
        segmentos = self._dividir_texto_en_segmentos(transcripcion, 3)
        
        for i, segmento in enumerate(segmentos):
            if i % 2 == 0:
                comentario = f"En esta parte del video se menciona: {segmento[:200]}..."
                intercambios.append((participantes[i % len(participantes)], comentario))
            else:
                analisis = self.resumir_texto(segmento, num_sentencias=2)
                intercambios.append((participantes[i % len(participantes)], f"Analizando esto: {analisis}"))
        
        # Agregar conclusión
        intercambios.append((participantes[1], f"En resumen, este video de {autor} ofrece una perspectiva interesante sobre el tema tratado."))
        
        # Generar audio
        audios = []
        for participante, texto in intercambios:
            archivo_audio = self.audio_manager.texto_a_audio_avanzado(
                texto, participante.lower().split()[0], 'es'
            )
            if archivo_audio:
                audios.append(archivo_audio)
        
        # Formatear conversación
        conversacion_texto = f"**🎬 Conversación sobre: {titulo}**\n\n"
        conversacion_texto += "\n\n".join([f"**{p}:** {t}" for p, t in intercambios])
        
        return conversacion_texto, audios

    def _dividir_texto_en_segmentos(self, texto: str, num_segmentos: int) -> List[str]:
        """Divide el texto en segmentos para análisis"""
        palabras = texto.split()
        longitud_segmento = len(palabras) // num_segmentos
        
        segmentos = []
        for i in range(num_segmentos):
            inicio = i * longitud_segmento
            fin = (i + 1) * longitud_segmento if i < num_segmentos - 1 else len(palabras)
            segmento = ' '.join(palabras[inicio:fin])
            if segmento.strip():
                segmentos.append(segmento)
        
        return segmentos

    def iniciar_servidor_web(self, host='localhost', port=5000):
        """Inicia el servidor web para chat interactivo"""
        print("🌐 Iniciando servidor web para chat interactivo...")
        self.web_chat_server.iniciar_servidor(host, port)

    def iniciar_interfaz_gradio(self, share=False):
        """Inicia la interfaz web con Gradio"""
        if not WEB_CHAT_AVAILABLE:
            print("Gradio no está disponible. Instala: pip install gradio")
            return
        
        interface = self.gradio_interface.crear_interfaz_gradio()
        if interface:
            print("🚀 Iniciando interfaz Gradio...")
            interface.launch(share=share, server_name="0.0.0.0" if share else "127.0.0.1")

    def buscar_contenido_web(self, consulta: str, num_resultados: int = None) -> Tuple[str, List[str], List[str]]:
        """Busca contenido web con soporte mejorado"""
        if num_resultados is None:
            num_resultados = self.configuracion['num_resultados_busqueda']
        
        cache_key = f"web_{consulta}_{num_resultados}"
        resultado_cache = self.cache_busquedas.get(cache_key)
        if resultado_cache and self.feedback.get(consulta, "útil") == "útil":
            return resultado_cache
        
        enlaces = []
        textos = []
        titulos = []
        
        try:
            from googlesearch import search
            logger.info(f"Buscando información sobre: {consulta}")
            
            # Buscar URLs
            urls = list(search(consulta, num_results=num_resultados*2, lang="es"))
            urls_seguras = [url for url in urls if URLValidator.es_url_segura(url)][:num_resultados]
            
            if not urls_seguras:
                logger.warning("No se encontraron URLs seguras")
                return "", [], []
            
            logger.info(f"Procesando {len(urls_seguras)} URLs seguras...")
            
            # Procesar URLs en paralelo
            with ThreadPoolExecutor(max_workers=3) as executor:
                resultados = list(executor.map(self._procesar_url_mejorado, urls_seguras))
            
            # Procesar resultados
            for resultado in resultados:
                if resultado:
                    url, titulo, texto = resultado
                    enlaces.append(url)
                    titulos.append(titulo)
                    textos.append(texto)
                    
        except Exception as e:
            logger.error(f"Error en la búsqueda: {e}")
            return "", [], []
        
        texto_completo = " ".join(textos)
        resultado = (texto_completo, enlaces, titulos)
        self.cache_busquedas.set(cache_key, resultado)
        return resultado

    def _procesar_url_mejorado(self, url: str) -> Optional[Tuple[str, str, str]]:
        """Procesa una URL y extrae su contenido con soporte mejorado"""
        for intento in range(self.configuracion['reintentos_url']):
            try:
                if not URLValidator.es_url_segura(url):
                    logger.warning(f"URL no segura detectada: {url}")
                    return None
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none'
                }
                
                try:
                    import requests
                    resp = requests.get(
                        url, 
                        timeout=self.configuracion['timeout_request'], 
                        headers=headers, 
                        allow_redirects=True, 
                        stream=True,
                        verify=True
                    )
                    resp.raise_for_status()
                    
                    # Verificar tipo de contenido
                    content_type = resp.headers.get('content-type', '').lower()
                    if 'text/html' not in content_type and 'text/plain' not in content_type:
                        logger.warning(f"Tipo de contenido no soportado: {content_type}")
                        return None
                    
                    # Verificar tamaño del contenido
                    content_length = resp.headers.get('content-length')
                    if content_length and int(content_length) > 2*1024*1024:  # 2MB max
                        logger.warning(f"Contenido demasiado grande: {content_length} bytes")
                        return None
                    
                except (Timeout, ConnectionError, RequestException) as e:
                    if intento == self.configuracion['reintentos_url'] - 1:
                        logger.warning(f"Error final procesando {url} después de {intento+1} intentos: {e}")
                        return None
                    else:
                        logger.info(f"Reintentando {url} (intento {intento+1})")
                        time.sleep(1)
                        continue
                
                # Procesar contenido HTML
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Extraer título
                titulo_elem = soup.find('title')
                titulo = titulo_elem.get_text().strip() if titulo_elem else url.split('/')[-1]
                titulo = self._limpiar_titulo(titulo)
                
                # Limpiar elementos no deseados
                for elemento in soup(["script", "style", "nav", "header", "footer", "aside", 
                                    "advertisement", "iframe", "object", "embed", "form", "noscript"]):
                    elemento.decompose()
                
                # Extraer contenido principal
                contenido = self._extraer_contenido_principal(soup)
                
                if not ValidadorContenido.es_contenido_valido(contenido, url):
                    logger.warning(f"Contenido inválido detectado para {url}")
                    return None
                
                return (url, titulo, contenido) if contenido else None
                
            except Exception as e:
                if intento == self.configuracion['reintentos_url'] - 1:
                    logger.warning(f"Error final procesando {url}: {e}")
                    return None
                else:
                    time.sleep(0.5)
                    continue
        return None

    def _limpiar_titulo(self, titulo: str) -> str:
        """Limpia y normaliza títulos"""
        titulo = re.sub(r'[^\w\s\-áéíóúñüÁÉÍÓÚÑÜ]', '', titulo)
        
        palabras_remover = [
            '- YouTube', '| Facebook', '- Twitter', '| Instagram', 
            '- Wikipedia', '| Google', '- Bing', '| Yahoo', '- Reddit',
            '| LinkedIn', '- Medium', '| Stack Overflow'
        ]
        
        for palabra in palabras_remover:
            titulo = titulo.replace(palabra, '')
        
        titulo = re.sub(r'\s+', ' ', titulo).strip()
        
        if len(titulo) > 100:
            titulo = titulo[:97] + "..."
        
        return titulo

    def _extraer_contenido_principal(self, soup) -> str:
        """Extrae el contenido principal de una página web"""
        selectors = [
            'main', 'article', '[role="main"]', '.main-content', '.content', 
            '.post-content', '.entry-content', '#content', '#main-content', 
            '#post-content', '#main', '.article-content', '.story-body',
            '.post-body', '.content-body'
        ]
        
        contenido_principal = None
        
        for selector in selectors:
            elementos = soup.select(selector)
            for elemento in elementos:
                texto = elemento.get_text().strip()
                if len(texto) > 200:
                    contenido_principal = elemento
                    break
            if contenido_principal:
                break
        
        if not contenido_principal:
            contenido_principal = soup.find('body') or soup
        
        elementos_texto = contenido_principal.find_all([
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'span', 'blockquote'
        ])
        
        textos = []
        for elem in elementos_texto:
            texto = elem.get_text().strip()
            if len(texto) > 30 and not self._es_texto_spam(texto):
                textos.append(texto)
            
            if len(textos) >= 50:
                break
        
        return ' '.join(textos)

    def _es_texto_spam(self, texto: str) -> bool:
        """Detecta texto spam o irrelevante"""
        spam_patterns = [
            r'cookies?', r'privacidad', r'términos', r'condiciones', r'suscríb',
            r'newsletter', r'publicidad', r'anunci', r'facebook', r'twitter',
            r'instagram', r'seguir', r'compartir', r'me gusta', r'like',
            r'click here', r'more info', r'read more', r'continue reading'
        ]
        
        texto_lower = texto.lower()
        spam_count = sum(1 for pattern in spam_patterns if re.search(pattern, texto_lower))
        
        return spam_count > 2
