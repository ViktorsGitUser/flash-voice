"""python -m flashvoice  - the page and the ear on FV_HOST:FV_PORT (default http://127.0.0.1:8099)."""
import ipaddress
import socket

import uvicorn

from . import config


def self_signed() -> tuple[str, str]:
    """A self-signed certificate for this computer's addresses (certs/), so a phone in the same network may use the
    microphone (browsers allow it only over https). The phone shows a warning once; a real certificate (Tailscale
    serve, Caddy ...) avoids that."""
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    d = config.ROOT / "certs"
    crt, key = d / "flash-voice.crt", d / "flash-voice.key"
    if crt.exists() and key.exists():
        return str(crt), str(key)
    d.mkdir(exist_ok=True)
    ips = {"127.0.0.1"}
    try:
        ips |= {i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        pass
    k = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "flash-voice")])
    now = datetime.datetime.now(datetime.timezone.utc)
    alt = [x509.DNSName("localhost"), x509.DNSName(socket.gethostname())] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now).not_valid_after(now + datetime.timedelta(days=825))
         .add_extension(x509.SubjectAlternativeName(alt), False).sign(k, hashes.SHA256()))
    crt.write_bytes(c.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    print(f"[flash-voice] self-signed certificate for {', '.join(sorted(ips))} in {d}", flush=True)
    return str(crt), str(key)


def main():
    ssl = {}
    if config.CERT == "self":
        crt, key = self_signed()
        ssl = {"ssl_certfile": crt, "ssl_keyfile": key}
    elif config.CERT:
        ssl = {"ssl_certfile": config.CERT, "ssl_keyfile": config.KEY}
    if config.HOST not in ("127.0.0.1", "localhost") and not config.PASSWORD:
        print("[flash-voice] WARNING: reachable from other devices without FV_PASSWORD - anyone in the network can "
              "use your LLM and microphone page", flush=True)
    scheme = "https" if ssl else "http"
    print(f"[flash-voice] {scheme}://{config.HOST}:{config.PORT}/  LLM {config.LLM_URL}  TTS {config.TTS_URL}  "
          f"STT {config.STT_URL}  Clef {config.CLEF_URL or '-'}", flush=True)
    uvicorn.run("flashvoice.app:app", host=config.HOST, port=config.PORT, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
