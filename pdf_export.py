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
        """Setup custom paragraph styles with consistent formatting"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#F47521'),
            spaceAfter=20,
            spaceBefore=0,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=28
        ))
        
        # Section heading style (## headings)
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#F47521'),
            spaceAfter=6,
            spaceBefore=16,
            fontName='Helvetica-Bold',
            leading=18
        ))
        
        # Subheading style (### or **bold:** lines)
        self.styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#333333'),
            spaceAfter=4,
            spaceBefore=10,
            fontName='Helvetica-Bold',
            leading=16
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#222222'),
            spaceAfter=6,
            spaceBefore=0,
            alignment=TA_LEFT,
            fontName='Helvetica',
            leading=14
        ))
        
        # Bullet point style (indented)
        self.styles.add(ParagraphStyle(
            name='CustomBullet',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#222222'),
            spaceAfter=4,
            spaceBefore=0,
            leftIndent=16,
            alignment=TA_LEFT,
            fontName='Helvetica',
            leading=14
        ))
        
        # Nested bullet style
        self.styles.add(ParagraphStyle(
            name='CustomNestedBullet',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#444444'),
            spaceAfter=3,
            spaceBefore=0,
            leftIndent=32,
            alignment=TA_LEFT,
            fontName='Helvetica',
            leading=14
        ))
        
        # Code style
        self.styles.add(ParagraphStyle(
            name='CustomCode',
            parent=self.styles['Code'],
            fontSize=8,
            textColor=colors.HexColor('#333333'),
            backColor=colors.HexColor('#F5F5F5'),
            spaceAfter=8,
            spaceBefore=4,
            fontName='Courier',
            leading=11
        ))
        
        # Metadata style (date, user info)
        self.styles.add(ParagraphStyle(
            name='CustomMeta',
            parent=self.styles['BodyText'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            spaceAfter=4,
            fontName='Helvetica',
            leading=12
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
        elements.append(Spacer(1, 0.15*inch))
        
        # Metadata
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        metadata = f"<b>Generated for:</b> {user_email}&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;<b>Date:</b> {timestamp}"
        elements.append(Paragraph(metadata, self.styles['CustomMeta']))
        elements.append(Spacer(1, 0.15*inch))
        
        # Divider line
        elements.append(self._create_divider())
        
        return elements
    
    def _create_query_section(self, query: str) -> List:
        """Create query section"""
        elements = []
        
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("Query", self.styles['CustomHeading']))
        
        # Clean and format query text
        query_clean = query.replace('<', '&lt;').replace('>', '&gt;')
        elements.append(Paragraph(f"<i>{query_clean}</i>", self.styles['CustomBody']))
        elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _create_response_section(self, response_text: str) -> List:
        """Create response section with full markdown rendering"""
        elements = []
        
        elements.append(Paragraph("Analysis &amp; Insights", self.styles['CustomHeading']))
        
        # Parse markdown content
        parsed_elements = self._parse_markdown(response_text)
        elements.extend(parsed_elements)
        
        return elements
    
    def _safe_paragraph(self, text: str, style) -> 'Paragraph':
        """Create a Paragraph, falling back to plain text if XML parsing fails."""
        try:
            return Paragraph(text, style)
        except Exception:
            # Strip all XML/HTML tags and try again with plain text
            clean = re.sub(r'<[^>]+>', '', text)
            try:
                return Paragraph(clean, style)
            except Exception:
                return Paragraph("(content could not be rendered)", style)

    def _parse_markdown(self, text: str) -> List:
        """Parse markdown text and convert to ReportLab elements"""
        elements = []
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                i += 1
                continue
            
            # Headers (## or ###)
            if stripped.startswith('###'):
                header_text = stripped.lstrip('#').strip()
                header_text = self._convert_markdown_inline(header_text)
                elements.append(self._safe_paragraph(header_text, self.styles['CustomSubheading']))
                i += 1
                continue
            
            if stripped.startswith('##'):
                header_text = stripped.lstrip('#').strip()
                header_text = self._convert_markdown_inline(header_text)
                elements.append(self._safe_paragraph(header_text, self.styles['CustomHeading']))
                i += 1
                continue
            
            # Bold section headers like **Summary:**
            if stripped.startswith('**') and ':' in stripped and stripped.endswith('**'):
                header_text = stripped.strip('*').strip()
                header_text = self._convert_markdown_inline(header_text)
                elements.append(self._safe_paragraph(f"<b>{header_text}</b>", self.styles['CustomSubheading']))
                i += 1
                continue
            
            # Nested bullet points (indented with spaces/tabs)
            if (line.startswith('    ') or line.startswith('\t')) and stripped.startswith(('-', '*', '•', '▪', '■')):
                nested_text = stripped.lstrip('-*•▪■').strip()
                nested_text = self._convert_markdown_inline(nested_text)
                elements.append(self._safe_paragraph(f"◦ {nested_text}", self.styles['CustomNestedBullet']))
                i += 1
                continue
            
            # Bullet points (-, *, •)
            if stripped.startswith(('-', '*', '•')) and not stripped.startswith('**'):
                bullet_text = stripped.lstrip('-*•').strip()
                bullet_text = self._convert_markdown_inline(bullet_text)
                elements.append(self._safe_paragraph(f"• {bullet_text}", self.styles['CustomBullet']))
                i += 1
                continue
            
            # Numbered lists
            if re.match(r'^\d+\.\s', stripped):
                num = stripped.split('.')[0]
                list_text = re.sub(r'^\d+\.\s', '', stripped)
                list_text = self._convert_markdown_inline(list_text)
                elements.append(self._safe_paragraph(f"{num}. {list_text}", self.styles['CustomBullet']))
                i += 1
                continue
            
            # Regular paragraph
            para_text = self._convert_markdown_inline(stripped)
            elements.append(self._safe_paragraph(para_text, self.styles['CustomBody']))
            i += 1
        
        return elements
    
    def _convert_markdown_inline(self, text: str) -> str:
        """Convert inline markdown (bold, italic, code) to ReportLab markup.
        
        Order matters: extract code spans first so underscores inside
        code (e.g. `snapshot_date`) are not turned into <i> tags, which
        would break XML nesting inside <font> tags.
        """
        # Escape HTML first
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # 1. Extract inline code spans into placeholders to protect them
        code_spans = []
        def _stash_code(m):
            code_spans.append(m.group(1))
            return f"\x00CODE{len(code_spans) - 1}\x00"
        text = re.sub(r'`(.+?)`', _stash_code, text)
        
        # 2. Bold (**text** or __text__)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
        
        # 3. Italic (*text* or _text_) — only match when surrounded by whitespace/punctuation
        #    to avoid false positives on snake_case words like active_subscribers
        text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<i>\1</i>', text)
        
        # 4. Links [text](url) - just show text
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        
        # 5. Restore code spans
        for i, code in enumerate(code_spans):
            text = text.replace(f"\x00CODE{i}\x00", f'<font name="Courier" size=10>{code}</font>')
        
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
    
    @staticmethod
    def _humanize_column_name(col: str) -> str:
        """Convert technical column names to readable format.
        snake_case -> Title Case, remove common suffixes."""
        # Common abbreviation mappings
        abbreviations = {
            'fts': 'FTS',
            'dtp': 'DTP',
            'nns': 'NNS',
            'avod': 'AVOD',
            'svod': 'SVOD',
            'id': 'ID',
        }
        # Remove common technical prefixes/suffixes
        col = re.sub(r'^(total_|sum_|avg_|count_)', '', col)
        
        parts = col.replace('_', ' ').split()
        humanized = []
        for part in parts:
            lower = part.lower()
            if lower in abbreviations:
                humanized.append(abbreviations[lower])
            else:
                humanized.append(part.capitalize())
        return ' '.join(humanized)
    
    def _create_table_section(self, table_data: Dict[str, Any]) -> List:
        """Create table section with auto-fitting columns and humanized headers"""
        elements = []
        
        elements.append(Paragraph("Data Table", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.05*inch))
        
        columns = table_data.get('columns', [])
        data = table_data.get('data', [])
        
        if not columns or not data:
            return elements
        
        # Limit rows for PDF
        max_rows = 20
        total_rows = len(data)
        if len(data) > max_rows:
            data = data[:max_rows]
            elements.append(Paragraph(
                f"<i>Showing first {max_rows} of {total_rows} rows</i>",
                self.styles['CustomMeta']
            ))
        
        # Humanize column headers
        display_columns = [self._humanize_column_name(c) for c in columns]
        
        # Limit columns if too many (max 8 for readability)
        max_cols = 8
        if len(columns) > max_cols:
            columns = columns[:max_cols]
            display_columns = display_columns[:max_cols]
            elements.append(Paragraph(
                f"<i>Showing first {max_cols} of {len(table_data.get('columns', []))} columns</i>",
                self.styles['CustomMeta']
            ))
        
        # Format cell values — use M/K suffixes for large numbers, truncate long strings
        def format_cell(val):
            if val is None:
                return '-'
            # Convert numeric strings (including Databricks scientific notation like '2.3426056E7')
            if isinstance(val, str):
                s = val.strip()
                if re.match(r'^-?[0-9]*\.?[0-9]+([Ee][+-]?[0-9]+)?$', s):
                    try:
                        val = float(s)
                    except (ValueError, TypeError):
                        pass  # Fall through to string handling below
                else:
                    # Non-numeric string (date, name) — just truncate if very long
                    return s[:32] + '...' if len(s) > 35 else s
            if isinstance(val, (int, float)):
                abs_val = abs(float(val))
                if abs_val >= 1_000_000_000:
                    return f"{val / 1_000_000_000:.4g}B"
                elif abs_val >= 1_000_000:
                    return f"{val / 1_000_000:.4g}M"
                elif abs_val >= 1_000:
                    return f"{val / 1_000:.4g}K"
                elif abs_val >= 1:
                    if isinstance(val, float) and val != int(val):
                        return f"{val:.4g}"
                    return str(int(val))
                elif abs_val == 0.0:
                    return '0'
                else:
                    return f"{val:.4g}"
            s = str(val)
            return s[:32] + '...' if len(s) > 35 else s
        
        # Build table rows — handle both dict rows and list rows
        table_rows = [display_columns]
        for row in data:
            if isinstance(row, dict):
                table_rows.append([format_cell(row.get(c, '')) for c in columns])
            elif isinstance(row, (list, tuple)):
                table_rows.append([format_cell(row[i]) if i < len(row) else '-' for i in range(len(columns))])
            else:
                # Unexpected row type — skip
                continue
        
        # Calculate column widths to fit page (6.5 inches usable)
        usable_width = 6.5 * inch
        num_cols = len(columns)
        col_width = usable_width / num_cols
        # Ensure minimum readable width
        min_width = 0.7 * inch
        if col_width < min_width:
            col_width = min_width
        col_widths = [col_width] * num_cols
        
        # Determine font size based on column count
        if num_cols > 6:
            header_font_size = 7
            body_font_size = 7
        elif num_cols > 4:
            header_font_size = 8
            body_font_size = 7
        else:
            header_font_size = 9
            body_font_size = 8
        
        # Wrap cell text in Paragraphs for word-wrapping
        cell_style = ParagraphStyle(
            'TableCell', parent=self.styles['BodyText'],
            fontSize=body_font_size, fontName='Helvetica', leading=body_font_size + 2,
            textColor=colors.HexColor('#222222'), alignment=TA_CENTER
        )
        header_style = ParagraphStyle(
            'TableHeader', parent=self.styles['BodyText'],
            fontSize=header_font_size, fontName='Helvetica-Bold', leading=header_font_size + 3,
            textColor=colors.white, alignment=TA_CENTER
        )
        
        wrapped_rows = []
        # Header row
        wrapped_rows.append([Paragraph(str(h), header_style) for h in display_columns])
        # Data rows
        for row in table_rows[1:]:
            wrapped_rows.append([Paragraph(str(v), cell_style) for v in row])
        
        # Create table
        t = Table(wrapped_rows, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F47521')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), header_font_size),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FAFAFA')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FAFAFA'), colors.HexColor('#F0F0F0')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('FONTSIZE', (0, 1), (-1, -1), body_font_size),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _create_charts_section(self, charts: List[Dict[str, Any]]) -> List:
        """Create charts section"""
        elements = []
        
        if not charts:
            return elements
        
        elements.append(PageBreak())
        elements.append(Paragraph("Visualizations", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.1*inch))
        
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
        """Convert Plotly chart to PNG image bytes, applying the same formatting as the UI"""
        try:
            import plotly.io as pio
            import json
            import numpy as np

            # Recreate figure from JSON - handle both dict and string
            if isinstance(plotly_json, str):
                fig = pio.from_json(plotly_json, skip_invalid=True)
            else:
                fig = pio.from_json(json.dumps(plotly_json), skip_invalid=True)

            # ── Apply same fix_chart_formatting logic as app.py ──────────────
            if fig.data:
                # Detect date x-axis
                has_date_x = False
                if hasattr(fig.data[0], 'x') and fig.data[0].x is not None and len(fig.data[0].x) > 0:
                    x_sample = fig.data[0].x[0]
                    has_date_x = (
                        isinstance(x_sample, str) and any(sep in str(x_sample) for sep in ['-', '/', ':'])
                        or 'datetime' in str(type(x_sample)).lower()
                        or 'timestamp' in str(type(x_sample)).lower()
                    )

                for trace in fig.data:
                    if hasattr(trace, 'y') and trace.y is not None:
                        y_values = [float(v) if isinstance(v, str) else v for v in trace.y]
                        trace.y = y_values
                        if y_values:
                            y_min, y_max = min(y_values), max(y_values)
                            y_range = y_max - y_min
                            if y_range > 0:
                                padding = y_range * 0.05
                                fig.update_yaxes(range=[y_min - padding, y_max + padding])
                            else:
                                fig.update_yaxes(range=[y_min * 0.95, y_max * 1.05])

                    # Sort bars by value descending (non-date x-axis only)
                    if trace.type == 'bar' and not has_date_x and hasattr(trace, 'y') and hasattr(trace, 'x'):
                        try:
                            pairs = sorted(zip(trace.x, trace.y),
                                           key=lambda p: float(p[1]) if p[1] is not None else 0,
                                           reverse=True)
                            if pairs:
                                trace.x, trace.y = zip(*pairs)
                        except Exception:
                            pass

                # Axis tick formatting
                if has_date_x:
                    fig.update_xaxes(exponentformat='none')
                else:
                    fig.update_xaxes(tickformat=',.0f', exponentformat='none')
                fig.update_yaxes(tickformat=',.0f', exponentformat='none')

            # ── Apply Crunchyroll dark theme (matches UI exactly) ─────────────
            fig.update_layout(
                plot_bgcolor="#0B0B0B",
                paper_bgcolor="#0B0B0B",
                font=dict(color="#cccccc", size=13, family="Arial, sans-serif"),
                title_font=dict(color="#F47521", size=16, family="Arial, sans-serif"),
                margin=dict(l=60, r=30, t=50, b=50),
                xaxis=dict(
                    gridcolor="#222222",
                    zerolinecolor="#333333",
                    color="#cccccc",
                    title_font=dict(size=13, color="#F47521"),
                    tickfont=dict(size=11, color="#999999"),
                    showline=False,
                ),
                yaxis=dict(
                    gridcolor="#222222",
                    zerolinecolor="#333333",
                    color="#cccccc",
                    title_font=dict(size=13, color="#F47521"),
                    tickfont=dict(size=11, color="#999999"),
                    showline=False,
                ),
                legend=dict(font=dict(size=11, color="#cccccc")),
            )

            # Convert to PNG bytes using kaleido
            img_bytes = pio.to_image(fig, format='png', width=1000, height=600, engine='kaleido')
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
