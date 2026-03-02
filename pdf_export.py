"""PDF Export Module for Agent Responses
Generates comprehensive PDF reports with insights, recommendations, charts, and conversation history
"""
import io
import base64
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas


class PDFExporter:
    """Generate PDF reports from agent responses"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#F47521'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#F47521'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10,
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            textColor=colors.HexColor('#000000'),
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Code style
        self.styles.add(ParagraphStyle(
            name='CustomCode',
            parent=self.styles['Code'],
            fontSize=9,
            textColor=colors.HexColor('#333333'),
            backColor=colors.HexColor('#F5F5F5'),
            spaceAfter=12,
            fontName='Courier'
        ))
    
    def export_conversation(
        self,
        query: str,
        response_text: str,
        charts: List[Dict[str, Any]] = None,
        table_data: Dict[str, Any] = None,
        sql_query: str = None,
        user_email: str = "anonymous",
        conversation_history: List[Dict[str, Any]] = None
    ) -> bytes:
        """
        Export a complete conversation to PDF
        
        Args:
            query: User's query
            response_text: Agent's response
            charts: List of chart configurations
            table_data: Table data with columns and rows
            sql_query: SQL query (if any)
            user_email: User email
            conversation_history: Previous conversation history
            
        Returns:
            PDF bytes
        """
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Container for PDF elements
        story = []
        
        # Add header
        story.extend(self._create_header(user_email))
        
        # Add query section
        story.extend(self._create_query_section(query))
        
        # Add response section
        story.extend(self._create_response_section(response_text))
        
        # Add SQL section (if present)
        if sql_query:
            story.extend(self._create_sql_section(sql_query))
        
        # Add table section (if present)
        if table_data:
            story.extend(self._create_table_section(table_data))
        
        # Add charts section (if present)
        if charts:
            story.extend(self._create_charts_section(charts))
        
        # Add conversation history (if present)
        if conversation_history and len(conversation_history) > 0:
            story.append(PageBreak())
            story.extend(self._create_conversation_history_section(conversation_history))
        
        # Add footer
        story.extend(self._create_footer())
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _create_header(self, user_email: str) -> List:
        """Create PDF header"""
        elements = []
        
        # Title
        title = Paragraph("Analytics Agent Report", self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Metadata
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        metadata = f"<b>Generated for:</b> {user_email}<br/><b>Date:</b> {timestamp}"
        elements.append(Paragraph(metadata, self.styles['CustomBody']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Divider line
        elements.append(self._create_divider())
        
        return elements
    
    def _create_query_section(self, query: str) -> List:
        """Create query section"""
        elements = []
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Query", self.styles['CustomHeading']))
        
        # Clean and format query text
        query_clean = query.replace('<', '&lt;').replace('>', '&gt;')
        elements.append(Paragraph(query_clean, self.styles['CustomBody']))
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_response_section(self, response_text: str) -> List:
        """Create response section with full markdown rendering"""
        elements = []
        
        elements.append(Paragraph("Analysis & Insights", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Parse markdown content
        parsed_elements = self._parse_markdown(response_text)
        elements.extend(parsed_elements)
        
        return elements
    
    def _parse_markdown(self, text: str) -> List:
        """Parse markdown text and convert to ReportLab elements"""
        elements = []
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Headers (## or **Section:**)
            if line.startswith('##'):
                header_text = line.lstrip('#').strip()
                header_text = self._convert_markdown_inline(header_text)
                elements.append(Paragraph(header_text, self.styles['CustomSubheading']))
                elements.append(Spacer(1, 0.1*inch))
                i += 1
                continue
            
            # Bold section headers like **Summary:**
            if line.startswith('**') and ':' in line:
                header_text = line.strip('*').strip()
                header_text = self._convert_markdown_inline(header_text)
                elements.append(Paragraph(header_text, self.styles['CustomSubheading']))
                elements.append(Spacer(1, 0.08*inch))
                i += 1
                continue
            
            # Bullet points (-, *, •)
            if line.startswith(('-', '*', '•')):
                bullet_text = line.lstrip('-*•').strip()
                bullet_text = self._convert_markdown_inline(bullet_text)
                elements.append(Paragraph(f"• {bullet_text}", self.styles['CustomBody']))
                i += 1
                continue
            
            # Numbered lists
            if re.match(r'^\d+\.\s', line):
                list_text = re.sub(r'^\d+\.\s', '', line)
                list_text = self._convert_markdown_inline(list_text)
                elements.append(Paragraph(f"{line.split('.')[0]}. {list_text}", self.styles['CustomBody']))
                i += 1
                continue
            
            # Nested bullet points with indentation
            if line.startswith('    ') or line.startswith('\t'):
                nested_text = line.strip()
                if nested_text.startswith(('-', '*', '•', '▪', '■')):
                    nested_text = nested_text.lstrip('-*•▪■').strip()
                    nested_text = self._convert_markdown_inline(nested_text)
                    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;◦ {nested_text}", self.styles['CustomBody']))
                    i += 1
                    continue
            
            # Regular paragraph
            para_text = self._convert_markdown_inline(line)
            elements.append(Paragraph(para_text, self.styles['CustomBody']))
            elements.append(Spacer(1, 0.08*inch))
            i += 1
        
        return elements
    
    def _convert_markdown_inline(self, text: str) -> str:
        """Convert inline markdown (bold, italic, code) to ReportLab markup"""
        # Escape HTML
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Bold (**text** or __text__)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
        
        # Italic (*text* or _text_)
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)
        
        # Inline code (`text`)
        text = re.sub(r'`(.+?)`', r'<font name="Courier" size=10>\1</font>', text)
        
        # Links [text](url) - just show text
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        
        return text
    
    def _create_sql_section(self, sql_query: str) -> List:
        """Create SQL section"""
        elements = []
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("SQL Query", self.styles['CustomHeading']))
        
        # Format SQL
        sql_clean = sql_query.replace('<', '&lt;').replace('>', '&gt;')
        sql_para = Paragraph(f"<font name='Courier' size=9>{sql_clean}</font>", self.styles['CustomCode'])
        elements.append(sql_para)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_table_section(self, table_data: Dict[str, Any]) -> List:
        """Create table section"""
        elements = []
        
        elements.append(Paragraph("Data Table", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.1*inch))
        
        columns = table_data.get('columns', [])
        data = table_data.get('data', [])
        
        if not columns or not data:
            return elements
        
        # Limit rows for PDF
        max_rows = 20
        if len(data) > max_rows:
            data = data[:max_rows]
            elements.append(Paragraph(f"<i>Showing first {max_rows} of {len(table_data.get('data', []))} rows</i>", self.styles['CustomBody']))
        
        # Create table data
        table_rows = [columns] + data
        
        # Create table
        t = Table(table_rows, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F47521')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_charts_section(self, charts: List[Dict[str, Any]]) -> List:
        """Create charts section"""
        elements = []
        
        if not charts:
            return elements
        
        elements.append(PageBreak())
        elements.append(Paragraph("Visualizations", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        charts_added = 0
        for i, chart in enumerate(charts):
            try:
                # Get chart title
                chart_title = chart.get('title', f'Chart {i+1}')
                print(f"[PDF] Processing chart {i+1}: {chart_title}")
                
                # Convert Plotly chart to image
                plotly_json = chart.get('plotly_json')
                if plotly_json:
                    print(f"[PDF] Chart {i+1} has plotly_json data")
                    # Convert to image bytes
                    img_bytes = self._plotly_to_image(plotly_json)
                    if img_bytes:
                        print(f"[PDF] Successfully converted chart {i+1} to image ({len(img_bytes)} bytes)")
                        elements.append(Paragraph(chart_title, self.styles['CustomSubheading']))
                        # Add image to PDF
                        img = Image(io.BytesIO(img_bytes), width=6*inch, height=4*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 0.3*inch))
                        charts_added += 1
                    else:
                        print(f"[PDF] Failed to convert chart {i+1} to image")
                else:
                    print(f"[PDF] Chart {i+1} has no plotly_json data")
                
            except Exception as e:
                print(f"Error adding chart {i} to PDF: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"[PDF] Added {charts_added} charts to PDF")
        return elements
    
    def _plotly_to_image(self, plotly_json: Dict) -> Optional[bytes]:
        """Convert Plotly chart to PNG image bytes"""
        try:
            import plotly.io as pio
            import json
            
            # Recreate figure from JSON - handle both dict and string
            if isinstance(plotly_json, str):
                fig_dict = json.loads(plotly_json)
                fig = go.Figure(fig_dict)
            else:
                fig = go.Figure(plotly_json)
            
            # Convert to PNG bytes using kaleido
            img_bytes = pio.to_image(fig, format='png', width=900, height=600, engine='kaleido')
            return img_bytes
            
        except ImportError as e:
            print(f"Error: kaleido package not installed. Install with: pip install kaleido")
            print(f"Full error: {e}")
            return None
        except Exception as e:
            print(f"Error converting Plotly to image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_conversation_history_section(self, conversation_history: List[Dict[str, Any]]) -> List:
        """Create conversation history section"""
        elements = []
        
        elements.append(Paragraph("Conversation History", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        for i, exchange in enumerate(conversation_history):
            query = exchange.get('query', '')
            answer = exchange.get('answer', '')
            
            # Query
            elements.append(Paragraph(f"<b>Q{i+1}:</b> {query}", self.styles['CustomBody']))
            
            # Answer (truncated)
            answer_preview = answer[:500] + '...' if len(answer) > 500 else answer
            answer_clean = answer_preview.replace('<', '&lt;').replace('>', '&gt;')
            elements.append(Paragraph(f"<b>A{i+1}:</b> {answer_clean}", self.styles['CustomBody']))
            elements.append(Spacer(1, 0.15*inch))
            
            # Add divider
            elements.append(self._create_divider())
        
        return elements
    
    def _create_footer(self) -> List:
        """Create PDF footer"""
        elements = []
        
        elements.append(Spacer(1, 0.3*inch))
        elements.append(self._create_divider())
        
        footer_text = "<i>This report was generated by the Analytics Agent powered by Databricks Genie</i>"
        elements.append(Paragraph(footer_text, self.styles['CustomBody']))
        
        return elements
    
    def _create_divider(self):
        """Create a divider line"""
        return Table(
            [['']],
            colWidths=[6.5*inch],
            style=TableStyle([
                ('LINEABOVE', (0, 0), (-1, 0), 2, colors.HexColor('#F47521')),
            ])
        )


# Convenience function for Streamlit
def create_pdf_download_button(
    query: str,
    response_text: str,
    charts: List[Dict[str, Any]] = None,
    table_data: Dict[str, Any] = None,
    sql_query: str = None,
    user_email: str = "anonymous",
    conversation_history: List[Dict[str, Any]] = None,
    button_label: str = "📄 Download as PDF"
) -> bytes:
    """
    Create PDF and return bytes for Streamlit download button
    
    Returns:
        PDF bytes ready for download
    """
    exporter = PDFExporter()
    pdf_bytes = exporter.export_conversation(
        query=query,
        response_text=response_text,
        charts=charts,
        table_data=table_data,
        sql_query=sql_query,
        user_email=user_email,
        conversation_history=conversation_history
    )
    return pdf_bytes
