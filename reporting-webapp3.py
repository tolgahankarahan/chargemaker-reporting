import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import requests

# Set page configuration
st.set_page_config(
    page_title="Intern Reporting Webapp",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to process uploaded data
def process_data(df):
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Handle potential NaN or missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('').astype(str)
    
    try:
        # Convert to proper datatypes
        df['Gestartet'] = pd.to_datetime(df['Gestartet'], errors='coerce')
        df['Beendet'] = pd.to_datetime(df['Beendet'], errors='coerce')
        
        # Handle potential format issues in Verbrauch column
        df['Verbrauch (kWh)'] = df['Verbrauch (kWh)'].astype(str).str.replace(',', '.').replace('', '0')
        df['Verbrauch (kWh)'] = pd.to_numeric(df['Verbrauch (kWh)'], errors='coerce').fillna(0)
        
        # Convert Kosten column to numeric, handling '-' values
        df['Kosten'] = df['Kosten'].astype(str).replace('-', '0').str.replace(',', '.')
        df['Kosten'] = pd.to_numeric(df['Kosten'], errors='coerce').fillna(0)
        
        # Filter out rows with invalid dates
        df = df.dropna(subset=['Gestartet', 'Beendet'])
        
        # Extract date from Gestartet
        df['Datum'] = df['Gestartet'].dt.date
        
        # Create a day+hour column for time series
        df['Tag'] = df['Gestartet'].dt.strftime('%Y-%m-%d')
        
        # Create charging duration column in hours - handle potential errors
        df['Ladezeit'] = (df['Beendet'] - df['Gestartet']).dt.total_seconds() / 3600
        df['Ladezeit'] = df['Ladezeit'].fillna(0).clip(lower=0.001)  # Avoid division by zero
        
        # Calculate charging rate
        df['Laderate (kW)'] = df['Verbrauch (kWh)'] / df['Ladezeit']
        
        return df
    
    except Exception as e:
        st.error(f"Fehler bei der Datenverarbeitung: {str(e)}")
        st.stop()

# Function to load CSV from URL
def load_csv_from_url(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Try different separators
        try:
            df = pd.read_csv(io.StringIO(response.text), sep=';')
        except:
            try:
                df = pd.read_csv(io.StringIO(response.text), sep=',')
            except:
                df = pd.read_csv(io.StringIO(response.text), sep=None, engine='python')
        
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden der CSV von URL: {str(e)}")
        return None

# Main function
def main():
    st.title("⚡ Intern Reporting Webapp")
    
    # Get URL parameters from make.com
    query_params = st.query_params
    csv_url = query_params.get("csv_url", None)
    csv_data = query_params.get("csv_data", None)  # NEW: Direct CSV content
    start_date_param = query_params.get("start_date", None)
    end_date_param = query_params.get("end_date", None)
    
    df = None
    
    # Check if CSV data is provided directly via URL parameter
    if csv_data:
        st.sidebar.success("📥 CSV-Daten direkt geladen...")
        try:
            # Decode base64 if needed (make.com can send base64 encoded data)
            import base64
            try:
                csv_content = base64.b64decode(csv_data).decode('utf-8')
            except:
                csv_content = csv_data  # Already plain text
            
            # Try different separators
            try:
                df = pd.read_csv(io.StringIO(csv_content), sep=';')
            except:
                try:
                    df = pd.read_csv(io.StringIO(csv_content), sep=',')
                except:
                    df = pd.read_csv(io.StringIO(csv_content), sep=None, engine='python')
        except Exception as e:
            st.error(f"Fehler beim Verarbeiten der CSV-Daten: {str(e)}")
            st.stop()
    
    # Check if CSV URL is provided via URL parameter
    elif csv_url:
        st.sidebar.success("📥 CSV-Datei wird von URL geladen...")
        df = load_csv_from_url(csv_url)
        
        if df is None:
            st.error("Fehler beim Laden der CSV-Datei von der URL. Bitte überprüfen Sie die URL.")
            st.stop()
    else:
        # File upload widget (Fallback)
        st.sidebar.header("Daten hochladen")
        uploaded_file = st.sidebar.file_uploader("CSV-Datei hochladen", type=["csv"])
        
        # Also provide URL input option
        st.sidebar.markdown("---")
        st.sidebar.subheader("Oder von URL laden")
        url_input = st.sidebar.text_input("CSV-URL eingeben")
        
        if url_input:
            df = load_csv_from_url(url_input)
        elif uploaded_file is not None:
            try:
                # Read the uploaded CSV file - try different separators if needed
                try:
                    df = pd.read_csv(uploaded_file, sep=';')
                except:
                    try:
                        df = pd.read_csv(uploaded_file, sep=',')
                    except:
                        df = pd.read_csv(uploaded_file, sep=None, engine='python')
            except Exception as e:
                st.error(f"Fehler beim Verarbeiten der Datei: {str(e)}")
                st.stop()
    
    # Check if we have data to process
    if df is not None:
        try:
            # Check if required columns are present
            required_columns = ['Ladepunkt', 'Gestartet', 'Beendet', 'Verbrauch (kWh)', 'Kundengruppe']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Fehlende Spalten in der CSV-Datei: {', '.join(missing_columns)}")
                st.stop()
            
            # Process the data
            df = process_data(df)
            
            if len(df) == 0:
                st.warning("Nach der Datenverarbeitung sind keine gültigen Datensätze übrig. Bitte überprüfen Sie Ihre CSV-Datei.")
                st.stop()
            
            # Sidebar with filter options
            st.sidebar.header("Filter")
            
            # Date range selection
            min_date = df['Gestartet'].min().date()
            max_date = df['Gestartet'].max().date()
            
            # Use URL parameters if provided, otherwise use defaults
            if start_date_param and end_date_param:
                try:
                    default_start = pd.to_datetime(start_date_param).date()
                    default_end = pd.to_datetime(end_date_param).date()
                    
                    # Ensure dates are within valid range
                    default_start = max(min_date, min(default_start, max_date))
                    default_end = max(min_date, min(default_end, max_date))
                    
                    st.sidebar.info(f"📅 Datum aus URL: {default_start} bis {default_end}")
                except:
                    default_start = min_date
                    default_end = max_date
            else:
                default_start = min_date
                default_end = max_date
            
            date_range = st.sidebar.date_input(
                "Zeitraum auswählen",
                [default_start, default_end],
                min_value=min_date,
                max_value=max_date
            )
            
            # Handle single date selection
            start_date = date_range[0]
            end_date = date_range[1] if len(date_range) > 1 else date_range[0]
            
            # Filter by Ladepunkt
            ladepunkte = sorted(df['Ladepunkt'].unique())
            selected_ladepunkte = st.sidebar.multiselect(
                "Ladepunkte auswählen",
                ladepunkte,
                default=ladepunkte
            )
            
            # Filter by customer group
            kundengruppen = sorted(df['Kundengruppe'].unique())
            selected_kundengruppen = st.sidebar.multiselect(
                "Kundengruppen auswählen",
                kundengruppen,
                default=kundengruppen
            )
            
            # Apply filters with safe date comparison
            filtered_df = df[
                (df['Gestartet'].dt.date >= pd.to_datetime(start_date).date()) & 
                (df['Gestartet'].dt.date <= pd.to_datetime(end_date).date()) &
                (df['Ladepunkt'].isin(selected_ladepunkte)) &
                (df['Kundengruppe'].isin(selected_kundengruppen))
            ]
            
            # Check if we have data after filtering
            if len(filtered_df) == 0:
                st.warning("Keine Daten für den ausgewählten Zeitraum und Filter gefunden.")
                st.stop()
            
            # Dashboard layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_kwh = filtered_df['Verbrauch (kWh)'].sum()
                st.metric("Gesamtverbrauch", f"{total_kwh:.2f} kWh")
            
            with col2:
                total_cost = filtered_df['Kosten'].sum()
                st.metric("Gesamtkosten", f"{total_cost:.2f} €")
            
            with col3:
                session_count = len(filtered_df)
                st.metric("Anzahl Ladevorgänge", session_count)
            
            # Visualizations
            st.subheader("Verbrauch nach Ladepunkt")
            
            # Aggregated data by charging point
            ladepunkt_summary = filtered_df.groupby('Ladepunkt').agg({
                'Verbrauch (kWh)': 'sum',
                'Kosten': 'sum',
                'Ladepunkt': 'count'
            }).rename(columns={'Ladepunkt': 'Anzahl Vorgänge'}).reset_index()
            
            # Bar chart for consumption by charging point
            fig1 = px.bar(
                ladepunkt_summary, 
                x='Ladepunkt', 
                y='Verbrauch (kWh)',
                color='Ladepunkt',
                text='Verbrauch (kWh)',
                title="Gesamtverbrauch pro Ladepunkt",
                labels={'Verbrauch (kWh)': 'Verbrauch (kWh)', 'Ladepunkt': 'Ladepunkt'}
            )
            
            fig1.update_traces(texttemplate='%{text:.1f} kWh', textposition='outside')
            fig1.update_layout(height=500)
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Daily consumption over time
            st.subheader("Täglicher Verbrauch über Zeit")
            
            # Aggregate by day
            daily_consumption = filtered_df.groupby('Tag').agg({
                'Verbrauch (kWh)': 'sum',
                'Ladepunkt': 'count'
            }).rename(columns={'Ladepunkt': 'Anzahl Vorgänge'}).reset_index()
            
            # Line chart for daily consumption
            fig2 = px.line(
                daily_consumption,
                x='Tag',
                y='Verbrauch (kWh)',
                markers=True,
                title="Täglicher Verbrauch",
                labels={'Verbrauch (kWh)': 'Verbrauch (kWh)', 'Tag': 'Datum'}
            )
            
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Customer group analysis
            st.subheader("Verbrauch nach Kundengruppe")
            
            # Aggregate by customer group
            customer_summary = filtered_df.groupby('Kundengruppe').agg({
                'Verbrauch (kWh)': 'sum',
                'Kosten': 'sum',
                'Ladepunkt': 'count'
            }).rename(columns={'Ladepunkt': 'Anzahl Vorgänge'}).reset_index()
            
            # Pie chart for consumption by customer group
            fig3 = px.pie(
                customer_summary,
                values='Verbrauch (kWh)',
                names='Kundengruppe',
                title="Verbrauch nach Kundengruppe"
            )
            
            fig3.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig3, use_container_width=True)
            
            # Charging efficiency analysis
            st.subheader("Laderate-Analyse")
            
            # Remove outliers for valid analysis (laderate > 0 and < 50)
            efficiency_df = filtered_df[(filtered_df['Laderate (kW)'] > 0) & (filtered_df['Laderate (kW)'] < 50)]
            
            if len(efficiency_df) > 0:
                # Box plot of charging rates by charging point
                fig4 = px.box(
                    efficiency_df,
                    x='Ladepunkt',
                    y='Laderate (kW)',
                    color='Ladepunkt',
                    title="Ladeleistungsverteilung nach Ladepunkt",
                    labels={'Laderate (kW)': 'Laderate (kW)', 'Ladepunkt': 'Ladepunkt'}
                )
                
                fig4.update_layout(height=500)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("Keine gültigen Laderate-Daten für die Analyse verfügbar.")
                
            
            # Heatmap der Ladedauer nach Wochentag mit 20-Stunden-Grenze
            st.subheader("Heatmap der Ladedauer nach Wochentag")

            # Wochentag und Stunde extrahieren
            filtered_df['Wochentag'] = filtered_df['Gestartet'].dt.day_name()
            filtered_df['Stunde'] = filtered_df['Gestartet'].dt.hour

            # Stellen sicher, dass die Wochentage in der richtigen Reihenfolge sind
            wochentage_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            filtered_df['Wochentag_num'] = filtered_df['Wochentag'].map(lambda x: wochentage_order.index(x))

            # Deutsche Wochentagsnamen für die Anzeige
            de_wochentage = {
                'Monday': 'Montag',
                'Tuesday': 'Dienstag',
                'Wednesday': 'Mittwoch',
                'Thursday': 'Donnerstag',
                'Friday': 'Freitag',
                'Saturday': 'Samstag',
                'Sunday': 'Sonntag'
            }
            filtered_df['Wochentag_de'] = filtered_df['Wochentag'].map(de_wochentage)

            # Ladedauer auf 20 Stunden begrenzen für eine bessere Darstellung
            filtered_df_for_heatmap = filtered_df.copy()
            filtered_df_for_heatmap['Ladezeit'] = filtered_df_for_heatmap['Ladezeit'].clip(upper=20)

            # Aggregierte Daten für die Heatmap vorbereiten
            # Durchschnittliche Ladedauer nach Wochentag und Stunde
            heatmap_data = filtered_df_for_heatmap.groupby(['Wochentag_de', 'Stunde']).agg({
                'Ladezeit': 'mean'
            }).reset_index()

            # Pivot-Tabelle für die Heatmap erstellen
            heatmap_pivot = heatmap_data.pivot(
                index='Wochentag_de',
                columns='Stunde',
                values='Ladezeit'
            )

            # Sortieren der Wochentage in der richtigen Reihenfolge
            heatmap_pivot = heatmap_pivot.reindex([de_wochentage[day] for day in wochentage_order])

            # Heatmap erstellen
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=[f"{hour}:00" for hour in heatmap_pivot.columns],
                y=heatmap_pivot.index,
                colorscale='Blues',
                colorbar=dict(title='Durchschn. Ladedauer (h)', tickvals=[0, 5, 10, 15, 20], ticktext=['0h', '5h', '10h', '15h', '20h+']),
                zmin=0,
                zmax=20,
                hoverongaps=False
            ))

            fig_heatmap.update_layout(
                title="Durchschnittliche Ladedauer nach Wochentag und Uhrzeit (begrenzt auf 20h)",
                xaxis_title="Startzeit der Ladung",
                yaxis_title="Wochentag",
                height=500
            )

            # Heatmap anzeigen
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Zusätzlich eine Übersicht der durchschnittlichen Ladedauer pro Wochentag erstellen
            st.subheader("Durchschnittliche Ladedauer pro Wochentag")

            # Durchschnittliche Ladedauer und Anzahl der Ladevorgänge pro Wochentag berechnen
            wochentag_summary = filtered_df.groupby('Wochentag_de').agg({
                'Ladezeit': ['mean', 'count']
            }).reset_index()

            wochentag_summary.columns = ['Wochentag', 'Durchschn. Ladedauer (h)', 'Anzahl Ladevorgänge']

            # Sortieren der Wochentage in der richtigen Reihenfolge
            wochentag_summary['Wochentag_order'] = wochentag_summary['Wochentag'].map(
                {tag: idx for idx, tag in enumerate([de_wochentage[day] for day in wochentage_order])}
            )
            wochentag_summary = wochentag_summary.sort_values('Wochentag_order').drop('Wochentag_order', axis=1)

            # Durchschnittliche Ladedauer auf 2 Dezimalstellen runden
            wochentag_summary['Durchschn. Ladedauer (h)'] = wochentag_summary['Durchschn. Ladedauer (h)'].round(2)

            # Balkendiagramm erstellen
            fig_bar = px.bar(
                wochentag_summary,
                x='Wochentag',
                y='Durchschn. Ladedauer (h)',
                text='Durchschn. Ladedauer (h)',
                title="Durchschnittliche Ladedauer nach Wochentag",
                color='Anzahl Ladevorgänge',
                color_continuous_scale='Viridis',
                labels={'Durchschn. Ladedauer (h)': 'Ladedauer (h)', 'Wochentag': 'Wochentag'}
            )

            fig_bar.update_traces(texttemplate='%{text:.2f} h', textposition='outside')
            fig_bar.update_layout(height=400)

            # Balkendiagramm anzeigen
            st.plotly_chart(fig_bar, use_container_width=True)
        

            # Detailed data table
            st.subheader("Detaillierte Ladedaten")
            
            # Show filtered data
            st.dataframe(
                filtered_df[['Ladepunkt', 'Gestartet', 'Beendet', 'Standzeit', 'Verbrauch (kWh)', 
                           'Kosten', 'Kundengruppe']].sort_values('Gestartet', ascending=False),
                use_container_width=True
            )
            
            # Add download button for processed data
            csv = filtered_df.to_csv(sep=';', index=False).encode('utf-8')
            st.download_button(
                label="Gefilterte Daten herunterladen",
                data=csv,
                file_name='chargeme_gefilterte_daten.csv',
                mime='text/csv',
            )
        
        except Exception as e:
            st.error(f"Fehler beim Verarbeiten der Datei: {str(e)}")
            st.code(str(e), language="python")
    
    else:
        # Display instructions when no file is uploaded
        st.info("Bitte laden Sie eine CSV-Datei hoch, um die Analyse zu starten.")
        st.markdown("""
        ### CSV-Format:
        Die CSV-Datei sollte folgende Spalten enthalten:
        - Ladepunkt
        - Status
        - Gestartet (Datum und Uhrzeit)
        - Beendet (Datum und Uhrzeit)
        - Monat (MM/JJJJ)
        - Stop-Grund
        - Standzeit
        - Zählerstand Start (kWh)
        - Zählerstand Ende (kWh)
        - Verbrauch (kWh)
        - Vertrag
        - Kundengruppe
        - Grund für die Auffälligkeit
        - Kosten
        - Provider
        - Operator
        
        Die Datei sollte mit Semikolon (;) als Trennzeichen formatiert sein.
        
        ### Verwendung mit URL-Parametern:
        Sie können die App auch mit URL-Parametern aufrufen:
        
        ```
        https://ihre-app.streamlit.app/?csv_url=https://example.com/data.csv&start_date=2025-01-01&end_date=2025-03-31
        ```
        
        Parameter:
        - `csv_url`: URL zur CSV-Datei
        - `start_date`: Startdatum (Format: YYYY-MM-DD)
        - `end_date`: Enddatum (Format: YYYY-MM-DD)
        """)
        
        # Display a sample data structure
        st.subheader("Beispiel-Datenstruktur:")
        sample_data = pd.DataFrame({
            'Ladepunkt': ['DE*CM1*E1200338*AC002', 'DE*CM1*E1200338*AC001'],
            'Status': ['Beendet', 'Beendet'],
            'Gestartet': ['2025-03-31 08:54:05', '2025-03-31 08:15:02'],
            'Beendet': ['2025-03-31 12:01:46', '2025-03-31 18:13:13'],
            'Monat (MM/JJJJ)': ['2025-03-31 12:01:46', '2025-03-31 18:13:13'],
            'Stop-Grund': ['-', '-'],
            'Standzeit': ['3 Stunden 7 Minuten 41 Sekunden', '9 Stunden 58 Minuten 11 Sekunden'],
            'Zählerstand Start (kWh)': [2080.209, 2674.691],
            'Zählerstand Ende (kWh)': [2090.891, 2727.168],
            'Verbrauch (kWh)': ['10,682', '52,477'],
            'Vertrag': ['DE-CM1-C70592550-S', 'DE-CM1-C16558995-I'],
            'Kundengruppe': ['Dachser_Mitarbeiter_Flotte', 'Dachser_Mitarbeiter_Flotte'],
            'Grund für die Auffälligkeit': ['-', '-'],
            'Kosten': ['4,170', '20,470'],
            'Provider': ['DE*CM1', 'DE*CM1'],
            'Operator': ['DE*CM1', 'DE*CM1']
        })
        st.dataframe(sample_data)
        
        # Add option to download sample data
        sample_csv = sample_data.to_csv(sep=';', index=False).encode('utf-8')
        st.download_button(
            label="Beispiel-CSV herunterladen",
            data=sample_csv,
            file_name='chargeme_beispiel.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
