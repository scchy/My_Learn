# python3
# Create Date: 2026-09-14
# Author: Scc_hy
# Func: 智能问答助手
# =================================================================================

import os
import time
from datetime import datetime
from typing import Optional
from hello_agents.tools import MemoryTool, RAGTool  # noqa: E402


class PDFLearningAssistant:
    def __init__(self, user_id='default_user'):
        self.user_id = user_id
        self.session_id = f"session__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # init tools 
        self.memory_tool = MemoryTool(user_id=self.user_id)
        self.memory_tool.current_session_id = self.session_id
        self.rag_tool = RAGTool(rag_namespace=f"pdf_{user_id}")

        # stats
        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0
        }

        # current doc
        self.current_document = None 
        
    def load_document(self, pdf_path: str):
        """
        Load a PDF document and process its content for question answering.
        
        :param pdf_path: PDF文件路径
        :type pdf_path: str
        """
        # Load and process the PDF file
        if not os.path.exists(pdf_path):
            return {"success": False, "message": f"文件不存在: {pdf_path}"}
        
        start_time = time.time()
        # ragtool  markItDown -> intell chunck -> embedding -> store in vector db
        result = self.rag_tool.execute(
            {"action": "add_document", "file_path": pdf_path, "chunk_size": 1000, "chunck_overlap": 200}
        )
        process_time = time.time() - start_time
        
        if result.get("success", False):
            self.current_document = os.path.basename(pdf_path)
            self.stats["documents_loaded"] += 1
            
            # memory_tool add  
            self.memory_tool.execute({
                "action": "add",
                "content": f"Loaded document: {self.current_document}",
                "memory_type": "episodic", 
                "importance": 0.9,
                "event_type": "document_load",
                "session_id": self.session_id
            })
            return {
                "success": True,
                "message": f"加载成功！(耗时: {process_time:.1f}秒)",
                "document": self.current_document
            }
        return {
            "success": False,
            "message": f"加载失败: {result.get('error', '未知错误')}(耗时: {process_time:.1f}秒)"
        }

    def ask(self, question: str, enable_advanced_search: bool = True):
        """Generate an answer based on the loaded PDF content

        Args:
            question (str): user query
            enable_advanced_search (bool, optional): Whether to enable advanced search  (MQE + HyDE)
        """
        if not self.current_document:
            return "⚠️ 请先加载文档！"

        # Memort Tool: record the question to work-memory 
        self.memory_tool.execute({
            "action": "add",
            "content": f"User asked: {question}",
            "memory_type": "working", 
            "importance": 0.6,
            "session_id": self.session_id
        })
        
        # rag tools  enable advanced search (MQE + HyDE) to generate answer
        answer = self.rag_tool.execute({
            "action": "ask",
            "question": question,
            "enable_advanced_search": enable_advanced_search,
            "enable_mqe": enable_advanced_search,
            "enable_hyde": enable_advanced_search,
            "limit": 5
        })
        
        # Memort Tool: recore the answer to work-memory
        answer_summary = answer[:150] + ("..." if len(answer) > 150 else "")
        self.memory_tool.execute({
            "action": "add",
            "content": f"关于'{question}', 回答要点:  {answer_summary}",
            "memory_type": "episodic",   
            "event_type": "qa_interaction",
            "importance": 0.7,
            "session_id": self.session_id
        })
        
        self.stats["questions_asked"] += 1
        return answer

    def add_note(self, content: str, concept: Optional[str] = None):
        """Add a note to the memory tool

        Args:
            content (str): note content
            concept (Optional[str], optional): concept associated with the note. Defaults to None.
        """
        result = self.memory_tool.execute({
            "action": "add",
            "content": content,
            "memory_type": "semantic", 
            "importance": 0.8,
            "concept": concept or "general",
            "session_id": self.session_id
        })
        return result

    def recall(self, query: str, limit: int = 5):
        """Recall memories based on a query

        Args:
            query (str): search query
            limit (int, optional): number of results to return. Defaults to 5.
        """
        result = self.memory_tool.execute({
            "action": "search",
            "query": query,
            "limit": limit
        })
        return result

    def get_stats(self):
        """Get session statistics"""
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        
        return {
            "session_duration": f"{duration:.1f} seconds",
            "documents_loaded": self.stats["documents_loaded"],
            "questions_asked": self.stats["questions_asked"],
            "concepts_learned": self.stats["concepts_learned"],
            "current_document": self.current_document or "Not loaded"
        }

    def generate_report(self, save_to_file: Optional[str] = None):
        """Generate a session report"""
        memory_summary = self.memory_tool.execute({"action": "summary", "limit": 10})
        rag_stats = self.rag_tool.execute({"action": "stats"})
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        
        report = {
            "session_info":{
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": duration
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"]
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats
        }
        if save_to_file:
            report_file = f"learning_report_{self.session_id}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            report["report_file"] = report_file
        return report
