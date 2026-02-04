import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging

# Configuração básica de logs para mostrar no terminal o que está acontecendo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExtratorVagas:
    def __init__(self, base_url, qtd_paginas=3):
        """
        Inicializa o extrator com a URL alvo e quantas páginas queremos raspar.
        """
        self.base_url = base_url
        self.qtd_paginas = qtd_paginas
        self.dados_coletados = []
        
        # Lista de User-Agents para 'enganar' o site e fingir que somos navegadores diferentes
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ]

    def _obter_html(self, url):
        """
        Função privada que faz a requisição ao site com segurança e pausa.
        """
        headers = {'User-Agent': random.choice(self.user_agents)}
        
        try:
            # Pausa aleatória entre 1 e 3 segundos para não ser bloqueado (Anti-Bot)
            time.sleep(random.uniform(1, 3))
            
            resposta = requests.get(url, headers=headers, timeout=10)
            resposta.raise_for_status() # Levanta erro se a página não carregar (ex: 404, 500)
            
            return BeautifulSoup(resposta.text, 'html.parser')
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao acessar {url}: {e}")
            return None

    def raspar_vagas(self):
        """
        Loop principal que percorre as páginas e extrai os dados.
        """
        logging.info(f"Iniciando a coleta de {self.qtd_paginas} páginas...")

        for pagina in range(1, self.qtd_paginas + 1):
            # Monta a URL (Atenção: verifique como o site faz paginação. Ex: ?page=1 ou /p/1)
            url_atual = f"{self.base_url}?page={pagina}" 
            logging.info(f"Acessando: {url_atual}")
            
            soup = self._obter_html(url_atual)
            
            if soup:
                # --- AQUI É ONDE A MÁGICA ACONTECE (AJUSTAR SELETORES CSS) ---
                # Exemplo genérico: Procura todas as 'divs' que têm classe de 'vaga'
                cartoes_vaga = soup.find_all('div', class_='cell-list-content') # Exemplo do Programathor
                
                if not cartoes_vaga:
                    logging.warning(f"Nenhuma vaga encontrada na página {pagina}. Verifique os seletores.")
                
                for cartao in cartoes_vaga:
                    try:
                        # Extração dos campos (Use Inspecionar Elemento no navegador para achar as classes)
                        titulo = cartao.find('h3').get_text(strip=True) if cartao.find('h3') else "Sem Título"
                        link = cartao.find('a')['href'] if cartao.find('a') else ""
                        
                        # Se o link for relativo (/vaga/123), adiciona o domínio
                        if link and not link.startswith('http'):
                            link = 'https://programathor.com.br' + link
                            
                        # Opcional: Entrar no link da vaga para pegar descrição completa (Etapa avançada)
                        # descricao_completa = self._pegar_detalhe_vaga(link)
                        
                        self.dados_coletados.append({
                            'titulo': titulo,
                            'link': link,
                            'origem': 'Programathor', # Exemplo
                            'data_coleta': time.strftime("%Y-%m-%d")
                        })
                    except Exception as e:
                        logging.error(f"Erro ao processar uma vaga específica: {e}")
                        continue

    def salvar_csv(self, nome_arquivo='vagas_brutas.csv'):
        """
        Salva os dados coletados em um arquivo CSV para o Rabay usar.
        """
        if not self.dados_coletados:
            logging.warning("Nenhum dado para salvar.")
            return
            
        df = pd.DataFrame(self.dados_coletados)
        df.to_csv(nome_arquivo, index=False, encoding='utf-8')
        logging.info(f"Sucesso! {len(df)} vagas salvas em '{nome_arquivo}'.")

# --- BLOCO DE EXECUÇÃO ---
if __name__ == "__main__":
    # Exemplo usando o Programathor (que é amigável para iniciantes)
    url_alvo = "https://programathor.com.br/jobs"
    
    bot = ExtratorVagas(base_url=url_alvo, qtd_paginas=2)
    bot.raspar_vagas()
    bot.salvar_csv("dados_para_rabay.csv")