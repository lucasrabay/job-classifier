import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ExtratorVagasWWR:
    """
    Scraper de vagas do We Work Remotely (WWR) usando BeautifulSoup.
    - Lista: https://weworkremotely.com/remote-full-time-jobs?page=2  (paginação via ?page=) :contentReference[oaicite:3]{index=3}
    - Links típicos de vagas: /remote-jobs/... :contentReference[oaicite:4]{index=4}
    """

    DOMINIO = "https://weworkremotely.com"

    def __init__(self, base_url, qtd_paginas=2, max_vagas=200):
        self.base_url = base_url
        self.qtd_paginas = qtd_paginas
        self.max_vagas = max_vagas
        self.dados_coletados = []

        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36',
        ]

    def _obter_html(self, url):
        headers = {'User-Agent': random.choice(self.user_agents)}

        try:
            time.sleep(random.uniform(1, 2.5))
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, 'html.parser')

        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao acessar {url}: {e}")
            return None

    def _normalizar_url(self, href):
        if not href:
            return ""
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return self.DOMINIO + href
        return self.DOMINIO + "/" + href

    def _extrair_links_listagem(self, soup):
        """
        Estratégia robusta:
        - pega todos <a> cujo href começa com /remote-jobs/
        - remove duplicados
        """
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("/remote-jobs/"):
                links.add(self._normalizar_url(href))
        return list(links)

    def _extrair_jsonld_jobposting(self, soup):
        """
        Procura JSON-LD (schema.org JobPosting) na página da vaga.
        Muitos sites incluem isso para SEO/Google Jobs. :contentReference[oaicite:5]{index=5}
        """
        scripts = soup.find_all("script", type="application/ld+json")
        for sc in scripts:
            txt = (sc.string or "").strip()
            if not txt:
                continue

            # Pode vir como objeto ou lista; às vezes vem com lixo de whitespace.
            try:
                data = json.loads(txt)
            except Exception:
                # fallback: tenta limpar (bem simples)
                txt2 = re.sub(r"\s+", " ", txt)
                try:
                    data = json.loads(txt2)
                except Exception:
                    continue

            # Pode ser lista de objetos
            candidates = data if isinstance(data, list) else [data]
            for obj in candidates:
                if isinstance(obj, dict) and obj.get("@type") == "JobPosting":
                    return obj

        return None

    def _pegar_detalhe_vaga(self, url_vaga):
        soup = self._obter_html(url_vaga)
        if not soup:
            return {}

        job = self._extrair_jsonld_jobposting(soup)

        if job:
            # Extrai campos comuns do JobPosting
            org = job.get("hiringOrganization", {}) if isinstance(job.get("hiringOrganization"), dict) else {}
            job_loc = job.get("jobLocation", [])
            if isinstance(job_loc, dict):
                job_loc = [job_loc]

            def fmt_loc(loc):
                if not isinstance(loc, dict):
                    return ""
                addr = loc.get("address", {})
                if not isinstance(addr, dict):
                    return ""
                parts = [addr.get("addressLocality", ""), addr.get("addressRegion", ""), addr.get("addressCountry", "")]
                return ", ".join([p for p in parts if p])

            locations = [fmt_loc(l) for l in job_loc]
            locations = ", ".join([l for l in locations if l])

            return {
                "titulo": job.get("title", ""),
                "empresa": org.get("name", ""),
                "descricao_texto": BeautifulSoup(job.get("description", ""), "html.parser").get_text(" ", strip=True),
                "data_publicacao": job.get("datePosted", ""),
                "tipo_emprego": job.get("employmentType", ""),
                "localizacao": locations,
            }

        # Fallback: pega um “miolo” de texto se não tiver JSON-LD
        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        main_text = soup.get_text(" ", strip=True)
        return {
            "titulo": title,
            "empresa": "",
            "descricao_texto": main_text[:2000],  # limita pra não explodir o CSV
            "data_publicacao": "",
            "tipo_emprego": "",
            "localizacao": "",
        }

    def raspar_vagas(self):
        logging.info(f"Iniciando coleta WWR: {self.qtd_paginas} páginas (máx {self.max_vagas} vagas)...")
        urls_vagas = []

        # 1) Coleta links das páginas de listagem
        for pagina in range(1, self.qtd_paginas + 1):
            url_atual = f"{self.base_url}?page={pagina}"
            logging.info(f"Acessando listagem: {url_atual}")

            soup = self._obter_html(url_atual)
            if not soup:
                continue

            links = self._extrair_links_listagem(soup)
            logging.info(f"Encontrados {len(links)} links de vagas na página {pagina}")
            urls_vagas.extend(links)

            if len(urls_vagas) >= self.max_vagas:
                break

        # dedup
        urls_vagas = list(dict.fromkeys(urls_vagas))[: self.max_vagas]
        logging.info(f"Total de links únicos de vagas: {len(urls_vagas)}")

        # 2) Visita cada vaga e extrai detalhes (HTML + BS4)
        for i, url_vaga in enumerate(urls_vagas, start=1):
            logging.info(f"[{i}/{len(urls_vagas)}] Coletando detalhe: {url_vaga}")

            detalhe = self._pegar_detalhe_vaga(url_vaga)
            if not detalhe:
                continue

            self.dados_coletados.append({
                "titulo": detalhe.get("titulo", ""),
                "empresa": detalhe.get("empresa", ""),
                "tipo_emprego": detalhe.get("tipo_emprego", ""),
                "localizacao": detalhe.get("localizacao", ""),
                "data_publicacao": detalhe.get("data_publicacao", ""),
                "descricao_texto": detalhe.get("descricao_texto", ""),
                "link": url_vaga,
                "origem": "WeWorkRemotely",
                "data_coleta": time.strftime("%Y-%m-%d"),
            })

    def salvar_csv(self, nome_arquivo="dados_para_rabay.csv"):
        if not self.dados_coletados:
            logging.warning("Nenhum dado para salvar.")
            return

        df = pd.DataFrame(self.dados_coletados)
        df.to_csv(nome_arquivo, index=False, encoding="utf-8")
        logging.info(f"Sucesso! {len(df)} vagas salvas em '{nome_arquivo}'.")


if __name__ == "__main__":
    # Fonte com paginação confirmada (?page=2) :contentReference[oaicite:6]{index=6}
    url_alvo = "https://weworkremotely.com/remote-full-time-jobs"

    bot = ExtratorVagasWWR(base_url=url_alvo, qtd_paginas=2, max_vagas=120)
    bot.raspar_vagas()
    bot.salvar_csv("data.csv")
