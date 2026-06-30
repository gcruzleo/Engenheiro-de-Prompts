import streamlit as st
import os
import base64
import tempfile
from groq import Groq
from st_copy_to_clipboard import st_copy_to_clipboard
import PyPDF2
import docx

# Configuração da página - minimalista e focada em leitura
st.set_page_config(page_title="Arquiteto de Prompts Sênior", layout="centered")

# Constantes
MODEL_NAME = "llama-3.3-70b-versatile"

# Inicialização do Cliente Groq
@st.cache_resource
def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY", "")
        
    if not api_key:
        st.error("Chave de API não configurada. Configure a variável GROQ_API_KEY nos secrets.")
        st.stop()
        
    return Groq(api_key=api_key)

client = get_groq_client()

# --- Processadores de Arquivos (Opção C) ---
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def process_file_attachment(uploaded_file, groq_client):
    """
    Processa arquivos PDF, DOCX, TXT, Áudio (Whisper) ou Imagens (Llama Vision).
    Retorna uma string formatada com o conteúdo para o contexto.
    """
    if uploaded_file is None:
        return ""
        
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    
    try:
        # 1. Documentos de Texto
        if "pdf" in file_type:
            return f"\n--- CONTEÚDO DO PDF ({file_name}) ---\n" + extract_text_from_pdf(uploaded_file)
        
        elif "wordprocessingml.document" in file_type or "docx" in file_name:
            return f"\n--- CONTEÚDO DO DOCX ({file_name}) ---\n" + extract_text_from_docx(uploaded_file)
            
        elif "text/plain" in file_type:
            return f"\n--- CONTEÚDO DO TXT ({file_name}) ---\n" + uploaded_file.getvalue().decode("utf-8")
            
        # 2. Áudio (Whisper API)
        elif "audio" in file_type or file_name.lower().endswith(('.mp3', '.wav', '.m4a')):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_name.split('.')[-1]}") as temp_audio:
                temp_audio.write(uploaded_file.getvalue())
                temp_audio_path = temp_audio.name
                
            try:
                with open(temp_audio_path, "rb") as f:
                    transcription = groq_client.audio.transcriptions.create(
                      file=(file_name, f.read()),
                      model="whisper-large-v3",
                      response_format="text"
                    )
                return f"\n--- TRANSCRIÇÃO DO ÁUDIO ({file_name}) ---\n" + transcription
            finally:
                os.remove(temp_audio_path)
                
        # 3. Imagens (Llama Vision API)
        elif "image" in file_type:
            base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Faça uma descrição extremamente detalhada desta imagem. Descreva os objetos, as pessoas, as cores, o ambiente, o estilo de iluminação, a técnica visual e qualquer texto presente. Esta descrição será usada como contexto para eu criar um prompt generativo depois."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{file_type};base64,{base64_image}",
                                }
                            }
                        ]
                    }
                ],
                model="llama-3.2-11b-vision-preview",
            )
            image_desc = chat_completion.choices[0].message.content
            return f"\n--- DESCRIÇÃO DA IMAGEM ({file_name}) ---\n[A IA de visão analisou a imagem anexada e relatou o seguinte:]\n" + image_desc
            
        else:
            st.warning(f"Formato ignorado: {file_type}. Apenas textos, áudios e imagens são suportados.")
            return ""
            
    except Exception as e:
        st.error(f"Erro ao processar {file_name}: {str(e)}")
        return ""

# --- Funções Dinâmicas de Prompt baseadas no Tipo de IA ---
def get_system_prompt_triagem(tipo_ia):
    return f"""Você é um analista investigativo de Engenharia de Prompts.
Sua missão é ler a ideia embrionária do usuário (e possivelmente o contexto de arquivos anexados) e formular de 2 a 3 perguntas cruciais para criar um "Rich Prompt" otimizado para: {tipo_ia}.

Técnica (Chain-of-Thought): Antes de gerar as perguntas, analise mentalmente quais parâmetros críticos estão faltando na ideia original (ex: falta de persona, formato, restrições).
As perguntas devem extrair exatamente essas lacunas, focando em:
- Objetivo principal e Contexto.
- Detalhes técnicos, estilísticos ou parâmetros específicos (dependendo da IA escolhida).
- Restrições estritas e nível de complexidade desejado.

As perguntas devem ser curtas, claras, objetivas e numeradas.
NÃO adicione saudações, introduções ou conclusões. Apenas liste as perguntas."""

def get_system_prompt_sintese(tipo_ia):
    base_prompt = "Atue como um Arquiteto de Prompts Sênior. Sua missão é transformar o input do usuário em um 'Rich Prompt' hiper-otimizado.\n"
    
    if "IA Geral" in tipo_ia:
        base_prompt += """
O FORMATO DE SAÍDA OBRIGATÓRIO deve ser construído estritamente sob o Framework C-P-C-F. O prompt final gerado DEVE ser uma ORDEM IMPERATIVA contendo as seguintes sessões claramente delimitadas em Markdown:
1. # Contexto: O substrato informacional fornecido.
2. # Persona: O papel que a IA alvo deve assumir (ex: "Atue como um Especialista Sênior em...").
3. # Comando: A tarefa atômica desambiguada.
4. # Formato e Restrições: Exigência do formato de saída e âncoras de 'Groundedness' (ex: "Responda apenas com base no contexto. Se não souber, diga 'não sei'").
5. # Raciocínio (CoT): Uma instrução obrigando a IA a "pensar passo a passo" antes de gerar a saída final.
O prompt não deve parecer um resumo, mas sim um microambiente de execução direto e imperativo."""
    elif "Vibecoding" in tipo_ia:
        base_prompt += """
O FORMATO DE SAÍDA OBRIGATÓRIO deve ser um documento Markdown robusto focado em agentes autônomos de código (Antigravity, Cursor). O prompt gerado DEVE conter:
1. # Contexto e Arquitetura: Explicação do problema.
2. # Stack Tecnológica: Exigência para o agente usar a stack explícita e o paradigma de programação correto.
3. # Plano de Implementação (CoT): Instruções granulares e modulares (Decomposição DAG).
4. # Restrições e Regras Estritas: Exigência categórica de código limpo (SOLID), documentação (JSDoc/Docstring), tratamento robusto de exceções e proibição de código preguiçoso.
O prompt DEVE ser escrito como uma instrução determinística para o agente de IA executar."""
    elif "Imagens" in tipo_ia:
        base_prompt += """
O FORMATO DE SAÍDA OBRIGATÓRIO deve ser EXCLUSIVAMENTE em formato JSON (.json) válido, otimizado para Modelos de Difusão Latente.
Estruture o JSON com chaves em inglês:
{
  "subject": "Objeto principal e contexto",
  "environment": "Ambiente e fundo",
  "lighting": "Iluminação e fotografia",
  "camera_parameters": "Lente, ângulo, profundidade de campo",
  "style": "Estilo artístico, renderização (ex: Unreal Engine)",
  "negative_prompt": "Parâmetros negativos rigidos (ex: mutated hands:1.5, blurry, text)",
  "raw_prompt": "String única juntando os valores positivos separados por vírgulas"
}"""
    elif "Vídeos" in tipo_ia:
        base_prompt += """
O FORMATO DE SAÍDA OBRIGATÓRIO deve ser EXCLUSIVAMENTE em formato JSON (.json) válido, otimizado para Modelos de Difusão Temporal (Sora, Runway).
Estruture o JSON com chaves em inglês:
{
  "subject_and_action": "Sujeito e sua ação detalhada",
  "camera_motion": "Direções de câmera (pan, tilt, zoom, tracking)",
  "scene_dynamics": "Coerência temporal e física do movimento",
  "lighting_and_atmosphere": "Iluminação cinematográfica, textura, clima",
  "negative_prompt": "Parâmetros negativos (ex: morphing, inconsistent framing, static)",
  "raw_prompt": "String única juntando os valores positivos pronta para o gerador"
}"""
        
    base_prompt += "\n\nRetorne APENAS o texto do prompt final (ou o JSON puro) pronto para uso. Não escreva introdução, conclusão ou comentários adicionais. Use blocos de código apropriados (```markdown ou ```json)."
    return base_prompt

# --- Gerenciamento de Estado (State Management) ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'ideia_inicial' not in st.session_state:
    st.session_state.ideia_inicial = ""
if 'perguntas' not in st.session_state:
    st.session_state.perguntas = ""
if 'respostas' not in st.session_state:
    st.session_state.respostas = ""
if 'prompt_final' not in st.session_state:
    st.session_state.prompt_final = ""
if 'tipo_ia' not in st.session_state:
    st.session_state.tipo_ia = "IA Geral (Gemini, ChatGPT, Claude)"

def reset_app():
    st.session_state.step = 1
    st.session_state.ideia_inicial = ""
    st.session_state.perguntas = ""
    st.session_state.respostas = ""
    st.session_state.prompt_final = ""
    st.session_state.tipo_ia = "IA Geral (Gemini, ChatGPT, Claude)"

# --- Cabeçalho Principal ---
st.title("Arquiteto de Prompts Sênior")
st.markdown("Transforme solicitações básicas e arquivos em prompts magistrais otimizados para diversas ferramentas de IA.")
st.divider()

# ETAPA 1: TRIAGEM
if st.session_state.step == 1:
    st.subheader("Etapa 1: Triagem")
    st.markdown("Descreva a sua ideia embrionária e anexe arquivos (se houver) para iniciarmos o projeto.")
    
    opcoes_ia = [
        "IA Geral (Gemini, ChatGPT, Claude)", 
        "Vibecoding / Programação (Antigravity, Cursor, etc)", 
        "Geração de Imagens (Midjourney, DALL-E, etc)", 
        "Geração de Vídeos (Sora, Runway, etc)"
    ]
    
    tipo_ia = st.selectbox(
        "Para qual tipo de IA este prompt será destinado?",
        options=opcoes_ia,
        index=opcoes_ia.index(st.session_state.tipo_ia) if st.session_state.tipo_ia in opcoes_ia else 0
    )
    
    ideia = st.text_area("Ideia Inicial:", value=st.session_state.ideia_inicial, height=150, placeholder="Ex: Preciso de um script em python... / Uma foto baseada na imagem anexada...")
    
    uploaded_file = st.file_uploader("Anexar arquivo de contexto (Opcional - Textos, Áudios ou Imagens)", type=["pdf", "txt", "docx", "mp3", "wav", "m4a", "png", "jpg", "jpeg"])
    
    if st.button("Analisar Ideia", type="primary"):
        if ideia.strip() or uploaded_file is not None:
            ideia_completa = ideia
            
            # Se houver arquivo, o sistema "lê" e traduz para texto
            if uploaded_file is not None:
                with st.spinner("Processando o arquivo anexado pelas IAs auxiliares (isso pode levar alguns segundos)..."):
                    conteudo_extraido = process_file_attachment(uploaded_file, client)
                    if conteudo_extraido:
                        ideia_completa += f"\n\n{conteudo_extraido}"
            
            with st.spinner("Analisando o contexto e formulando perguntas investigativas..."):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": get_system_prompt_triagem(tipo_ia)},
                            {"role": "user", "content": ideia_completa}
                        ],
                        model=MODEL_NAME,
                        temperature=0.7,
                    )
                    st.session_state.perguntas = chat_completion.choices[0].message.content
                    st.session_state.ideia_inicial = ideia_completa # Salvamos a ideia combinada com o anexo
                    st.session_state.tipo_ia = tipo_ia
                    st.session_state.step = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro na comunicação com a API: {e}")
        else:
            st.warning("Por favor, digite alguma ideia ou anexe um arquivo antes de continuar.")

# ETAPA 2: ENTREVISTA
elif st.session_state.step == 2:
    st.subheader("Etapa 2: Entrevista (Refinamento)")
    st.markdown(f"**🎯 Alvo:** {st.session_state.tipo_ia}\n\nPor favor, responda às perguntas abaixo para extrairmos o contexto perfeito para este modelo.")
    
    st.info(st.session_state.perguntas)
    
    respostas = st.text_area("Suas Respostas:", value=st.session_state.respostas, height=200, placeholder="1. O contexto é...\n2. O estilo será...")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Voltar para Triagem"):
            st.session_state.respostas = respostas
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Gerar Prompt Final", type="primary"):
            if respostas.strip():
                st.session_state.respostas = respostas
                with st.spinner("Sintetizando o seu Prompt Magistral..."):
                    contexto_completo = (
                        f"IDEIA INICIAL (INCLUINDO ANEXOS LIDOS):\n{st.session_state.ideia_inicial}\n\n"
                        f"PERGUNTAS DA ENTREVISTA:\n{st.session_state.perguntas}\n\n"
                        f"RESPOSTAS DO USUÁRIO:\n{st.session_state.respostas}"
                    )
                    try:
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": get_system_prompt_sintese(st.session_state.tipo_ia)},
                                {"role": "user", "content": contexto_completo}
                            ],
                            model=MODEL_NAME,
                            temperature=0.7,
                        )
                        st.session_state.prompt_final = chat_completion.choices[0].message.content
                        st.session_state.step = 3
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro na comunicação com a API: {e}")
            else:
                st.warning("Por favor, forneça as respostas antes de gerar o prompt.")

# ETAPA 3: SÍNTESE
elif st.session_state.step == 3:
    st.subheader("Etapa 3: Síntese (O Prompt Magistral)")
    st.markdown(f"Aqui está o seu prompt perfeitamente estruturado para **{st.session_state.tipo_ia}**.")
    
    st.code(st.session_state.prompt_final, language="markdown")
    
    st_copy_to_clipboard(
        text=st.session_state.prompt_final,
        before_copy_label="📋 Copiar para Área de Transferência",
        after_copy_label="✅ Copiado!"
    )
    
    st.write("") # Espaçamento
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Refazer Entrevista"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Iniciar Novo Projeto", type="primary"):
            reset_app()
            st.rerun()
